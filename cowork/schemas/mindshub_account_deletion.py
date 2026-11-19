"""Wire shapes for the caller's MindsHub account deletion request."""

from cowork.schemas.base import CamelResponse, IsoDatetime


class MindsHubDeletionRequestView(CamelResponse):
    """The caller's open deletion request: when they asked, and the date by
    which MindsHub deletes the account."""

    requested_at: IsoDatetime
    delete_by: IsoDatetime
