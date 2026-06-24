from uuid import UUID

from cowork.schemas.base import CamelRequest


class ProjectCreateRequest(CamelRequest):
    name: str


class ProjectUpdateRequest(CamelRequest):
    name: str | None = None
    # Mark this project as the most-recently-selected one. The client's
    # selection is canonical for interactive use; the server records it (via
    # last_selected_at) only as the headless/scheduled-run fallback. Replaces
    # the former dual ``is_active`` flag.
    last_selected: bool | None = None
    # Organization metadata (server-side, follows the user across devices).
    pinned: bool | None = None
    sort_order: int | None = None
    archived: bool | None = None


class ProjectReorderRequest(CamelRequest):
    # Project ids in the desired display order; sort_order is assigned 0..n.
    project_ids: list[UUID]
