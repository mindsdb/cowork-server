"""A failed owner-row drop after an artifact delete names the project and the
artifact. Slugs are unique only within a project, so the slug alone does not
say which project kept a stale owner row."""
from __future__ import annotations

import logging

import pytest
from sqlalchemy.exc import OperationalError

from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.services import artifact_ownership as ownership
from test_artifact_permissions import (  # noqa: F401  (world, org_deployment and no_publish_side_effects are fixtures)
    _write,
    no_publish_side_effects,
    org_deployment,
    scoped,
    world,
)

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

DRIVER_TEXT = "server closed the connection unexpectedly"


async def test_a_failed_owner_drop_names_the_project_and_the_artifact(
    world, org_deployment, granted_product_permissions, no_publish_side_effects,
    monkeypatch, caplog, owned_logger,
):
    org_id, creator, _member, project, source = world
    local_id, folder = _write(source, "mine")

    def fail_the_drop(*_a, **_k):
        raise OperationalError("DELETE FROM shared_resource_attributions", {}, Exception(DRIVER_TEXT))

    monkeypatch.setattr(ownership, "forget_artifact_owner", fail_the_drop)
    logged = owned_logger(artifacts_ep.logger.name, level=logging.WARNING)
    with scoped(org_id, creator) as session:
        await artifacts_ep.delete_artifact_for_request(session, local_id, project_id=project.id)

    assert not folder.exists()
    [record] = [
        record for record in caplog.records
        if record.name == artifacts_ep.logger.name and record.levelno == logging.WARNING
    ]
    assert record.getMessage().startswith("Database operation failed: error_type=OperationalError ")
    assert (record.project_id, record.artifact_slug) == (str(project.id), "mine")
    output = logged.output()
    assert f"[Project:{project.id}][Artifact:'mine']: Database operation failed" in output
    assert DRIVER_TEXT not in output
