"""The Fernet keys the server seals stored secrets with.

Two sources, never both. With ``COWORK_KEY_FILE`` in the server's process
environment, the keys come from that file alone: one urlsafe-base64 Fernet key
per line, where the first line seals and every line opens (``MultiFernet``).
The server then never reads ``COWORK_MASTER_KEY`` or ``MASTER_KEY_PATH`` and
never makes a key. Without it, the legacy key applies: ``COWORK_MASTER_KEY``,
else the file at ``MASTER_KEY_PATH``, made on first use unless the deployment
was sealed under a key file (cowork.services.sealing.sealing_marker_present).
"""

import logging
import os
import stat
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet, MultiFernet

logger = logging.getLogger(__name__)

KEY_FILE_ENV = "COWORK_KEY_FILE"
# A key line is 44 bytes; a few hundred lines is far past any rotation.
MAX_KEY_FILE_BYTES = 64 * 1024


class KeyFileError(RuntimeError):
    """The key file is missing, unreadable or malformed, or sealed data exists
    while no key file is configured. The message names the cause and, for a
    malformed file, the line number. It never holds a key."""


class UnopenableSettingError(Exception):
    """A stored secret, a setting or a channel credential, that the server's
    key doesn't open. The message names its settings key, never a value."""

    def __init__(self, *, key: str) -> None:
        self.key = key
        super().__init__(f"The stored secret {key} doesn't open with the server's key.")


def key_file_path() -> Path | None:
    """The file ``COWORK_KEY_FILE`` names, read from the process environment
    only, so a line in cowork-home's ``.env`` can't set or move it."""
    value = os.environ.get(KEY_FILE_ENV, "")
    return Path(value) if value else None


def read_key_file(*, path: Path) -> tuple[Fernet, ...]:
    """Every key in ``path``, in order, so the first one seals. Blank lines are skipped."""
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise KeyFileError(f"{KEY_FILE_ENV} names {path}, which isn't a regular file.")
        if info.st_size > MAX_KEY_FILE_BYTES:
            raise KeyFileError(f"{KEY_FILE_ENV} names {path}, which is larger than a key file can be.")
        raw = path.read_bytes()
    except FileNotFoundError:
        raise KeyFileError(
            f"{KEY_FILE_ENV} names {path}, which doesn't exist. The host writes it before "
            "the stack starts; check that it ran and that the folder is bound into the container."
        ) from None
    except PermissionError:
        raise KeyFileError(f"{KEY_FILE_ENV} names {path}, which the server's user can't read.") from None
    except OSError as error:
        raise KeyFileError(f"{KEY_FILE_ENV} names {path}, which can't be read: {error.strerror}.") from None
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise KeyFileError(f"{path} isn't text: each line holds one Fernet key.") from None
    keys: list[Fernet] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            keys.append(Fernet(line))
        except ValueError:
            raise KeyFileError(
                f"{path} line {number} isn't a Fernet key: each line holds 32 bytes in urlsafe base64."
            ) from None
    if not keys:
        raise KeyFileError(f"{path} holds no key.")
    return tuple(keys)


@lru_cache
def sealing_keys() -> MultiFernet | None:
    """The key file's keys when ``COWORK_KEY_FILE`` is set, else None."""
    path = key_file_path()
    return None if path is None else MultiFernet(read_key_file(path=path))


def warn_ignored_key_settings(*, master_key: str, master_key_path_set: bool, code_sandbox: str) -> None:
    """With a key file set, say which legacy key settings it overrides, and that
    agent code reads the key file while the code sandbox is off. The values
    are never logged."""
    if sealing_keys() is None:
        return
    if master_key:
        logger.warning(
            "COWORK_MASTER_KEY is ignored while COWORK_KEY_FILE is set",
            extra={"event": "key_setting_ignored"},
        )
    if master_key_path_set:
        logger.warning(
            "MASTER_KEY_PATH is ignored while COWORK_KEY_FILE is set",
            extra={"event": "key_setting_ignored"},
        )
    if code_sandbox == "off":
        logger.warning(
            "COWORK_CODE_SANDBOX is off, so agent code can read the key file COWORK_KEY_FILE names",
            extra={"event": "key_file_readable_by_agent_code"},
        )


def _load_or_create_master_key() -> bytes:
    from cowork.common.settings.app_settings import get_app_settings
    settings = get_app_settings()
    # A directly-provided key (COWORK_MASTER_KEY) wins over the file. Stateless
    # / cloud deploys set it from a Secret so the key is stable across pod
    # restarts and replicas; without it each pod would generate its own file
    # key and lose access to data encrypted by a previous pod.
    if settings.master_key:
        return settings.master_key.strip().encode()
    key_path = Path(settings.master_key_path)
    if key_path.exists():
        return key_path.read_bytes().strip()
    # A new key beside data sealed under a key file would read every sealed
    # setting as unset, and leave a key on the disk the key file kept off it.
    from cowork.services.sealing import refuse_sealed_state_without_key_file

    refuse_sealed_state_without_key_file()
    key = Fernet.generate_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    key_path.chmod(0o600)
    return key


@lru_cache
def get_fernet() -> Fernet | MultiFernet:
    keys = sealing_keys()
    return keys if keys is not None else Fernet(_load_or_create_master_key())


def encrypt(plaintext: str) -> str:
    return get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    return get_fernet().decrypt(token.encode()).decode()
