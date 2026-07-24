from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet


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
    key = Fernet.generate_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    key_path.chmod(0o600)
    return key


@lru_cache
def get_fernet() -> Fernet:
    return Fernet(_load_or_create_master_key())


def encrypt(plaintext: str) -> str:
    return get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    return get_fernet().decrypt(token.encode()).decode()
