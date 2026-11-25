"""A self-hosted process's environment for tests: the deployment kind and the
credentials a self-hosted process requires (secret_files.REQUIRED_SELF_HOSTED_NAMES)."""

import os

from cowork.common.settings.secret_files import REQUIRED_SELF_HOSTED_NAMES, file_variable

# What the self-hosted stack's compose file passes, as stand-ins. The test
# session's own DATABASE_URI stays as conftest set it.
_STAND_INS = {
    "COWORK_AUTH_TOKEN": "self-hosted-test-token",
    "COWORK_PUBLISH_TOKEN": "mdb_self_hosted_test_publish_token",
    "MINDS_API_KEY": "mdb_self_hosted_test_publish_token",
}


def enable_self_hosted(monkeypatch) -> None:
    """Set COWORK_DEPLOYMENT_KIND=self-hosted with each required credential the
    environment lacks, as the stack's compose file gives them."""
    monkeypatch.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
    for name in REQUIRED_SELF_HOSTED_NAMES:
        if not os.environ.get(name) and not os.environ.get(file_variable(name=name)):
            monkeypatch.setenv(name, _STAND_INS[name])
