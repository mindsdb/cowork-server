from cowork.api.v1.endpoints import coding_runtime
from cowork.api.v1.endpoints.guards import require_local_tenancy


def test_runtime_router_is_fail_closed_until_tenant_service_resolution_exists() -> None:
    dependencies = [dependency.dependency for dependency in coding_runtime.router.dependencies]

    assert require_local_tenancy in dependencies
