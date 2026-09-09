from pydantic import AliasChoices, Field

from cowork.common.paths import cowork_home
from cowork.common.settings import Settings


class HermesHarnessSettings(Settings):
    root_dir: str = Field(
        default_factory=lambda: str(cowork_home() / "hermes"),
        validation_alias=AliasChoices("HERMES_ROOT_DIR", "HERMES_HOME"),
        description="Root directory for all Hermes data (skills, memory, sessions, config)",
    )  # HERMES_ROOT_DIR or HERMES_HOME
