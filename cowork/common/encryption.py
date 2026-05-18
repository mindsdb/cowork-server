from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet


def _load_or_create_master_key() -> bytes:
    from cowork.common.settings import get_app_settings
    key_path = Path(get_app_settings().master_key_path)
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
