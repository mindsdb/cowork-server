"""Bodies of requests Cowork refuses before any work starts."""
from pydantic import BaseModel, ConfigDict


class Refusal(BaseModel):
    """A refusal a client shows, with a code it can branch on.

    ``detail`` is one sentence the web UI shows as it is. It stays a string:
    the UI renders any other ``detail`` as "[object Object]".
    """

    model_config = ConfigDict(frozen=True)

    detail: str
    code: str


# A second question into a conversation whose turn is still answering. It is
# refused with 409 before anything reads or saves it.
TURN_IN_PROGRESS = Refusal(
    detail=(
        "Another question is still being answered in this conversation. "
        "Wait for it to finish, then send yours again."
    ),
    code="turn_in_progress",
)
