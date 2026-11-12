"""Credentials read from files that the process environment names.

A deployment that keeps its credentials out of the container's environment
sets ``<NAME>_FILE`` to the path of a file that holds the value, in place of
``<NAME>`` itself. The settings classes read those files through
``SecretFileSettingsSource``, which ranks above the environment and every
``.env`` file, so the server, each ``cowork.ops`` command and every other
process that loads settings read the same value.

Only the process environment can name a file. A ``<NAME>_FILE`` line in
cowork-home's ``.env`` is ignored, so a line written there can't point a
credential read at another file.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic_settings import EnvSettingsSource
from pydantic_settings.sources.utils import parse_env_vars

#: The settings a self-hosted process must get from its environment, as a
#: value or a file (require_process_secrets).
REQUIRED_SELF_HOSTED_NAMES: tuple[str, ...] = (
    "DATABASE_URI",
    "COWORK_AUTH_TOKEN",
    "COWORK_PUBLISH_TOKEN",
    "MINDS_API_KEY",
)

#: The settings a deployment may read from files. Each one's file is named by
#: the same variable with ``_FILE`` appended. The live-apps launch token is set
#: only where live apps are on, and the endpoint key only where the server calls
#: the model endpoint itself, so a self-hosted process requires neither.
SECRET_FILE_NAMES: tuple[str, ...] = (
    *REQUIRED_SELF_HOSTED_NAMES,
    "COWORK_LIVE_APPS_LAUNCH_TOKEN",
    "OPENAI_COMPATIBLE_API_KEY",
)


class SecretFileError(RuntimeError):
    """A credential file is named but can't be used, or a self-hosted process has no credential.

    Messages name the variable and the path, never the file's contents.
    """


def file_variable(*, name: str) -> str:
    """The variable that names ``name``'s file."""
    return f"{name}_FILE"


def read_secret_file(*, name: str) -> str | None:
    """The value in the file the process environment names for ``name``.

    Returns None when ``<name>_FILE`` is unset or empty. One trailing newline is
    dropped, so a file written with ``echo`` holds the same value as one written
    without it.

    Raises:
        SecretFileError: ``<name>`` and ``<name>_FILE`` are both set, or the file
            can't be read, isn't UTF-8 text, or holds nothing.
    """
    variable = file_variable(name=name)
    path = os.environ.get(variable, "")
    if not path:
        return None
    if os.environ.get(name):
        raise SecretFileError(f"{name} and {variable} are both set; set only one of them")
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise SecretFileError(f"{variable} names {path}, which can't be read: {exc.strerror}") from None
    except UnicodeDecodeError:
        raise SecretFileError(f"{variable} names {path}, which isn't UTF-8 text") from None
    value = text.removesuffix("\n")
    if not value:
        raise SecretFileError(f"{variable} names {path}, which is empty")
    return value


def process_secret(*, name: str) -> str:
    """``name``'s value from the process environment: its file's, else its own, else empty.

    Use it where code must not take a value from cowork-home's ``.env``, which
    the settings classes also read.
    """
    from_file = read_secret_file(name=name)
    if from_file is not None:
        return from_file
    return os.environ.get(name, "")


def require_process_secrets() -> None:
    """Stop unless the process environment gives every ``REQUIRED_SELF_HOSTED_NAMES`` entry a value or a file.

    A self-hosted server runs on Postgres with credentials its operator
    provides. Without them the settings would fall back to a SQLite file in
    cowork-home, a generated token, or a line in cowork-home's ``.env``.

    Raises:
        SecretFileError: naming each variable that has neither.
    """
    missing = [
        name
        for name in REQUIRED_SELF_HOSTED_NAMES
        if not os.environ.get(name) and not os.environ.get(file_variable(name=name))
    ]
    if missing:
        names = ", ".join(f"{name} or {file_variable(name=name)}" for name in missing)
        raise SecretFileError(
            f"COWORK_DEPLOYMENT_KIND=self-hosted needs {names} in the process environment"
        )


class SecretFileSettingsSource(EnvSettingsSource):
    """The environment source, over ``SECRET_FILE_NAMES`` alone, with each value read from its file.

    It parses names the way the environment source does, so aliases such as
    ``COWORK_AUTH_TOKEN`` and nested names such as ``DATABASE_URI`` reach the
    same fields. A name whose ``_FILE`` variable is unset gives nothing, and the
    sources after this one decide it as before.
    """

    def _load_env_vars(self) -> Mapping[str, str | None]:
        values: dict[str, str | None] = {}
        for name in SECRET_FILE_NAMES:
            value = read_secret_file(name=name)
            if value is not None:
                values[name] = value
        return parse_env_vars(values, self.case_sensitive, self.env_ignore_empty, self.env_parse_none_str)
