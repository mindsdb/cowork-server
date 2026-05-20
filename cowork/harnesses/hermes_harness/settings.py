from pathlib import Path

from pydantic import AliasChoices, Field

from cowork.common.settings import Settings


class HermesHarnessSettings(Settings):
    hermes_home: str = Field(
        default=str(Path.home() / ".cowork" / "hermes"),
        validation_alias=AliasChoices("HERMES_HOME", "hermes_home"),
        description="Root directory for all Hermes data (skills, memory, sessions, config)",
    )  # HERMES_HOME
