"""python -m cowork.ops.reseal: seal a local deployment's saved connections and settings under
COWORK_KEY_FILE, as turning sealing on and rotating its key do.

It checks every vault before it writes anything and stops at a link, a file with a second name or
a misnamed record; it opens every record and sealed row with the key file's keys and the legacy key
and stops at one none opens; then it seals everything under the key file's first key, records the
switch, removes leftover temporary files, and deletes the legacy key file last. A refused run
leaves every file, link target and row as it was.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from cryptography.fernet import Fernet, MultiFernet
from sqlmodel import Session, SQLModel, select

from cowork.common.encryption import get_fernet, sealing_keys
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.live_app_preview import LiveAppPreview
from cowork.models.setting import Setting
from cowork.ops import reseal
from cowork.services.connectors.sealed_vault import SEALED_FIELD, SealedDataVault
from cowork.services.sealing import SEALING_MARKER_KEY

PARTITION_A = "44e92829-e701-599b-88f9-69f5bf1165eb"
PARTITION_B = "5a1d2c3e-4f50-4617-8a9b-0c1d2e3f4a5b"
SECRET_A = "zx-canary-7781"
SECRET_SETTING = "zx-canary-7784"


@pytest.fixture
def deployment(monkeypatch, tmp_path, key_file, managed_dir):
    """A local deployment under the test's folder: its own database, a base vault, two people's
    partitions, the managed folder, a legacy .master_key and a key file."""
    uri = f"sqlite:///{tmp_path / 'cowork.db'}"
    legacy = tmp_path / "home" / ".master_key"
    legacy.parent.mkdir()
    legacy.write_bytes(Fernet.generate_key())
    monkeypatch.setenv("DATABASE_URI", uri)
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("MASTER_KEY_PATH", str(legacy))
    monkeypatch.delenv("COWORK_MASTER_KEY", raising=False)
    get_app_settings.cache_clear()
    sealing_keys.cache_clear()
    get_fernet.cache_clear()
    engine = get_engine(uri)
    SQLModel.metadata.create_all(engine)
    vaults = {
        "base": tmp_path / "vault",
        "a": tmp_path / "shared" / PARTITION_A / "data-vault",
        "b": tmp_path / "shared" / PARTITION_B / "data-vault",
        "managed": managed_dir.vault_dir,
    }
    yield {"engine": engine, "legacy": legacy, "key_file": key_file, "vaults": vaults, "root": tmp_path}
    get_app_settings.cache_clear()
    sealing_keys.cache_clear()
    get_fernet.cache_clear()


def plain_records(deployment) -> None:
    vaults = deployment["vaults"]
    LocalDataVault(vaults["a"]).save(
        "postgres", "sales", {"host": "db.example", "port": 5432, "password": SECRET_A, "_label": "Sales"},
        secure_keys=["password"],
    )
    LocalDataVault(vaults["b"]).save("postgres", "orders", {"password": "b-secret"}, secure_keys=["password"])
    LocalDataVault(vaults["base"]).save("http", "status", {"token": "base-token"})
    LocalDataVault(vaults["managed"]).save("mysql", "operator", {"password": "managed-secret"}, secure_keys=["password"])


def store_setting(deployment, *, key: str, value: str) -> None:
    with Session(deployment["engine"]) as session:
        session.add(Setting(key=key, value=value))
        session.commit()


def legacy_fernet(deployment) -> Fernet:
    return Fernet(deployment["legacy"].read_bytes())


def file_fernet(deployment) -> Fernet:
    return Fernet(deployment["key_file"].read_bytes().strip())


def run(deployment, *arguments: str) -> int:
    return reseal.main(["--managed-vault", str(deployment["vaults"]["managed"]), *arguments])


def tree(root: Path) -> dict[str, object]:
    """Every file under ``root`` by its bytes, and every link by its target."""
    found: dict[str, object] = {}
    for folder, folders, files in os.walk(root, followlinks=False):
        for name in folders + files:
            path = Path(folder) / name
            if path.is_symlink():
                found[str(path)] = ("link", os.readlink(path))
            elif path.is_file():
                found[str(path)] = path.read_bytes()
    return found


def marker_present(deployment) -> bool:
    with Session(deployment["engine"]) as session:
        return session.exec(select(Setting).where(Setting.key == SEALING_MARKER_KEY)).first() is not None


def test_turning_on_seals_every_record_setting_and_cookie_then_removes_the_old_key(deployment, capsys):
    plain_records(deployment)
    store_setting(deployment, key="anthropic_api_key", value=legacy_fernet(deployment).encrypt(SECRET_SETTING.encode()).decode())
    leftover = deployment["vaults"]["base"] / "http-status.tmp"
    leftover.write_text("{}")
    with Session(deployment["engine"]) as session:
        for position, cookie in enumerate((legacy_fernet(deployment).encrypt(b"cookie").decode(), "not-a-token")):
            session.add(LiveAppPreview(
                owner_partition_id=PARTITION_A, owner_user_id=str(uuid4()), owner_email="a@example.com",
                viewer_cookie=cookie, position=position, launch_id=uuid4(), origin="https://app.example",
                intent_hash=uuid4().hex, state="active", expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            ))
        session.commit()

    assert run(deployment) == 0

    printed = capsys.readouterr().out
    assert printed.splitlines() == [
        "Sealed 4 saved connection(s) in 4 vault(s), 1 setting(s) and channel credential(s), "
        "and 1 live-app preview cookie(s) under the key file's first key.",
        "Cleared 1 preview cookie(s) no key opened; those previews ask to sign in again.",
        "Removed 1 leftover temporary file(s).",
        f"Removed the old key file {deployment['legacy']}.",
    ]
    keys = MultiFernet([file_fernet(deployment)])
    sealed = SealedDataVault(deployment["vaults"]["a"], keys=keys)
    assert sealed.load("postgres", "sales") == {"host": "db.example", "port": 5432, "password": SECRET_A, "_label": "Sales"}
    on_disk = (deployment["vaults"]["a"] / "postgres-sales").read_text()
    assert SECRET_A not in on_disk and json.loads(on_disk)["fields"][SEALED_FIELD] == 1
    assert json.loads(on_disk)["secure_keys"] == ["password"]
    # A record saved with no secure_keys keeps none, so the connection API's name heuristic still applies.
    assert "secure_keys" not in json.loads((deployment["vaults"]["base"] / "http-status").read_text())
    assert SealedDataVault(deployment["vaults"]["managed"], keys=keys).load("mysql", "operator") == {"password": "managed-secret"}
    with Session(deployment["engine"]) as session:
        row = session.exec(select(Setting).where(Setting.key == "anthropic_api_key")).one()
        cookies = session.exec(select(LiveAppPreview).order_by(LiveAppPreview.position)).all()
    assert file_fernet(deployment).decrypt(row.value.encode()) == SECRET_SETTING.encode()
    assert file_fernet(deployment).decrypt(cookies[0].viewer_cookie.encode()) == b"cookie"
    assert cookies[1].viewer_cookie is None
    assert marker_present(deployment)
    assert not leftover.exists() and not deployment["legacy"].exists()
    assert SECRET_A not in printed and SECRET_SETTING not in printed


def test_a_second_run_changes_nothing_but_the_records_timestamps(deployment):
    plain_records(deployment)
    assert run(deployment) == 0
    keys = MultiFernet([file_fernet(deployment)])
    before = SealedDataVault(deployment["vaults"]["a"], keys=keys).read_record("postgres", "sales")

    assert run(deployment) == 0

    after = SealedDataVault(deployment["vaults"]["a"], keys=keys).read_record("postgres", "sales")
    assert after["fields"] == before["fields"] and after["created_at"] == before["created_at"]


def test_rotation_seals_under_the_new_key_and_the_old_one_opens_nothing_after(deployment):
    plain_records(deployment)
    assert run(deployment) == 0
    old = deployment["key_file"].read_bytes().strip()
    new = Fernet.generate_key()
    deployment["key_file"].write_bytes(new + b"\n" + old + b"\n")

    assert run(deployment) == 0
    deployment["key_file"].write_bytes(new + b"\n")
    assert run(deployment) == 0

    assert SealedDataVault(deployment["vaults"]["a"], keys=MultiFernet([Fernet(new)])).load("postgres", "sales")["password"] == SECRET_A
    with pytest.raises(Exception):
        SealedDataVault(deployment["vaults"]["a"], keys=MultiFernet([Fernet(old)])).load("postgres", "sales")


def _refused(deployment, capsys, *, names: str) -> None:
    before = tree(deployment["root"])

    assert run(deployment) == 1

    assert names in capsys.readouterr().err
    assert tree(deployment["root"]) == before
    assert not marker_present(deployment)


def test_a_link_to_another_persons_record_stops_it_before_any_write(deployment, capsys):
    plain_records(deployment)
    (deployment["vaults"]["a"] / "postgres-orders").symlink_to(deployment["vaults"]["b"] / "postgres-orders")

    _refused(deployment, capsys, names="postgres-orders: is a link")


def test_a_hard_link_to_another_persons_record_stops_it_before_any_write(deployment, capsys):
    plain_records(deployment)
    os.link(deployment["vaults"]["b"] / "postgres-orders", deployment["vaults"]["a"] / "postgres-orders")

    _refused(deployment, capsys, names="has a second name")


def test_a_planted_temporary_link_stops_it_before_any_write(deployment, capsys):
    plain_records(deployment)
    (deployment["vaults"]["a"] / "postgres-sales.tmp").symlink_to(deployment["vaults"]["b"] / "postgres-orders")

    _refused(deployment, capsys, names="postgres-sales.tmp: is a link")


def test_a_record_under_another_name_stops_it(deployment, capsys):
    plain_records(deployment)
    os.rename(deployment["vaults"]["a"] / "postgres-sales", deployment["vaults"]["a"] / "postgres-renamed")

    _refused(deployment, capsys, names="holds postgres/sales, whose file name differs")


def test_a_setting_no_key_opens_stops_it_before_any_write(deployment, capsys):
    plain_records(deployment)
    store_setting(deployment, key="anthropic_api_key", value=Fernet(Fernet.generate_key()).encrypt(b"lost").decode())

    _refused(deployment, capsys, names="setting anthropic_api_key (scope global)")


def test_without_a_key_file_it_changes_nothing(deployment, monkeypatch, capsys):
    plain_records(deployment)
    monkeypatch.delenv("COWORK_KEY_FILE")

    _refused(deployment, capsys, names="COWORK_KEY_FILE isn't set")
