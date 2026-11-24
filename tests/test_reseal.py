"""python -m cowork.ops.reseal: seal a local deployment's saved connections and settings under
COWORK_KEY_FILE, as turning sealing on and rotating its key do.

It checks every vault before it writes anything and stops at a link, a file with a second name, a
temporary name that isn't a regular file, a misnamed record or one that isn't a saved connection;
it opens every record and sealed row with the key file's keys and the legacy key and stops at one
none opens; then it seals everything under the key file's first key, records the switch, removes
leftover temporary files, and deletes the legacy key file last. A refused run exits 1 and leaves
every file, link target and row as it was. A run an error stops part-way exits 2 and keeps the
legacy key, and the same command again finishes the job.

The channel plugins fill the process-wide registry only in create_app, which other tests call, so
each channel test empties it first, as a fresh ``python -m cowork.ops.reseal`` finds it.
"""
from __future__ import annotations

import errno
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from cryptography.fernet import Fernet, MultiFernet
from sqlmodel import Session, SQLModel, select

from cowork.channels import registry as channel_registry
from cowork.common.encryption import get_fernet, sealing_keys
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.session import get_engine
from cowork.models.live_app_preview import LiveAppPreview
from cowork.models.setting import Setting
from cowork.ops import reseal
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.identity import VAULT_KEEP_SENTINEL
from cowork.services.connectors.sealed_vault import SEALED_FIELD, SealedDataVault
from cowork.services.sealing import SEALING_MARKER_KEY

PARTITION_A = "44e92829-e701-599b-88f9-69f5bf1165eb"
PARTITION_B = "5a1d2c3e-4f50-4617-8a9b-0c1d2e3f4a5b"
SECRET_A = "zx-canary-7781"
SECRET_SETTING = "zx-canary-7784"
SECRET_CHANNEL = "zx-canary-7785"
USER = "8d1f0b6e-2a4c-4e7f-9b3d-5c6a7e8f9012"
ORG = "1c2d3e4f-5a6b-4c7d-8e9f-0a1b2c3d4e5f"


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


def store_setting(deployment, *, key: str, value: str, user: str | None = None) -> None:
    scope = {} if user is None else {"scope": "user", "user_id": user, "org_id": ORG}
    with Session(deployment["engine"]) as session:
        session.add(Setting(key=key, value=value, **scope))
        session.commit()


def setting_value(deployment, *, key: str) -> str:
    with Session(deployment["engine"]) as session:
        return session.exec(select(Setting).where(Setting.key == key)).one().value


@pytest.fixture
def no_channels_loaded(monkeypatch):
    """The process-wide channel registry empty, as a fresh python -m cowork.ops.reseal finds it."""
    empty = channel_registry.PluginRegistry()
    monkeypatch.setattr(channel_registry, "default_registry", empty)
    yield empty


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


def test_a_second_run_seals_the_same_values_again(deployment):
    """Run again, it seals the same values again: each record's updated_at and every sealed value's
    ciphertext change, and everything opens as before."""
    plain_records(deployment)
    store_setting(deployment, key="anthropic_api_key", value=legacy_fernet(deployment).encrypt(SECRET_SETTING.encode()).decode())
    assert run(deployment) == 0
    keys = MultiFernet([file_fernet(deployment)])
    record_file = deployment["vaults"]["a"] / "postgres-sales"
    before = json.loads(record_file.read_text())
    setting_before = setting_value(deployment, key="anthropic_api_key")

    assert run(deployment) == 0

    after = json.loads(record_file.read_text())
    assert after["created_at"] == before["created_at"] and after["updated_at"] != before["updated_at"]
    assert after["fields"]["password"] != before["fields"]["password"]
    assert setting_value(deployment, key="anthropic_api_key") != setting_before
    assert SealedDataVault(deployment["vaults"]["a"], keys=keys).load("postgres", "sales")["password"] == SECRET_A
    assert file_fernet(deployment).decrypt(setting_value(deployment, key="anthropic_api_key").encode()) == SECRET_SETTING.encode()


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


def _refused(deployment, capsys, *, names: str, arguments: list[str] | None = None) -> str:
    """Run the command, which must refuse naming ``names`` and change nothing: every file, the key
    file among them, stays as it was, and no marker row is written. Returns what it printed."""
    before = tree(deployment["root"])

    assert (run(deployment) if arguments is None else reseal.main(arguments)) == 1

    printed = capsys.readouterr().err
    assert names in printed
    assert tree(deployment["root"]) == before
    assert deployment["key_file"].exists()
    assert not marker_present(deployment)
    return printed


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
    store_setting(deployment, key="anthropic_api_key", value=Fernet(Fernet.generate_key()).encrypt(b"lost").decode(), user=USER)

    printed = _refused(deployment, capsys, names="setting anthropic_api_key (global)")

    assert f"setting anthropic_api_key (user {USER} in {ORG})" in printed
    assert "A row no key you hold opens can be entered again, or deleted, with Cowork started as before; then run this again." in printed


def test_without_a_key_file_it_changes_nothing(deployment, monkeypatch, capsys):
    plain_records(deployment)
    monkeypatch.delenv("COWORK_KEY_FILE")

    _refused(deployment, capsys, names="COWORK_KEY_FILE isn't set")


def test_a_key_file_that_is_the_legacy_key_file_stops_it(deployment, monkeypatch, capsys):
    plain_records(deployment)
    monkeypatch.setenv("COWORK_KEY_FILE", str(deployment["legacy"]))

    _refused(deployment, capsys, names=f"COWORK_KEY_FILE names the legacy key file {deployment['legacy']}: copy the key")

    assert deployment["legacy"].exists()


def test_an_old_key_file_that_is_the_key_file_stops_it(deployment, capsys):
    plain_records(deployment)
    arguments = ["--managed-vault", str(deployment["vaults"]["managed"]), "--old-key-file", str(deployment["key_file"])]

    _refused(deployment, capsys, names="COWORK_KEY_FILE names the legacy key file", arguments=arguments)

    assert deployment["key_file"].exists()


# ── channel credentials ─────────────────────────────────────────────────────


def test_turning_on_seals_a_channel_credential_with_no_channel_loaded(deployment, no_channels_loaded):
    store_setting(deployment, key="channel.slack.bot_token", value=legacy_fernet(deployment).encrypt(SECRET_CHANNEL.encode()).decode())

    assert run(deployment) == 0

    assert file_fernet(deployment).decrypt(setting_value(deployment, key="channel.slack.bot_token").encode()) == SECRET_CHANNEL.encode()
    assert not deployment["legacy"].exists()
    assert no_channels_loaded.channel_types() == []
    assert channel_registry.get_registry() is no_channels_loaded


def test_rotation_seals_a_channel_credential_under_the_new_key(deployment, no_channels_loaded):
    old = deployment["key_file"].read_bytes().strip()
    store_setting(deployment, key="channel.slack.bot_token", value=Fernet(old).encrypt(SECRET_CHANNEL.encode()).decode())
    assert run(deployment) == 0
    new = Fernet.generate_key()
    deployment["key_file"].write_bytes(new + b"\n" + old + b"\n")

    assert run(deployment) == 0
    deployment["key_file"].write_bytes(new + b"\n")
    assert run(deployment) == 0

    assert Fernet(new).decrypt(setting_value(deployment, key="channel.slack.bot_token").encode()) == SECRET_CHANNEL.encode()


def test_a_credential_of_a_channel_plugin_that_failed_to_import_stops_it(deployment, monkeypatch, capsys, no_channels_loaded):
    """load_first_party_plugins logs a plugin it can't import and goes on, so the Slack credentials
    drop out of the sealed set, and only the legacy key the run would delete opens this row."""
    monkeypatch.setitem(sys.modules, "cowork.channels.plugins.slack", None)
    store_setting(deployment, key="channel.slack.bot_token", value=legacy_fernet(deployment).encrypt(SECRET_CHANNEL.encode()).decode())

    printed = _refused(deployment, capsys, names="setting channel.slack.bot_token (global)")

    assert "open with the legacy key or an older key in the key file, not with its first key" in printed
    assert "A channel plugin that failed to import" in printed
    assert SECRET_CHANNEL not in printed


def test_a_credential_of_a_channel_plugin_that_failed_to_import_stops_a_rotation(deployment, monkeypatch, capsys, no_channels_loaded):
    """In a rotation the row opens with the key file's older key, which the operator drops next,
    and with neither the first key nor any legacy key."""
    old = deployment["key_file"].read_bytes().strip()
    deployment["key_file"].write_bytes(Fernet.generate_key() + b"\n" + old + b"\n")
    deployment["legacy"].unlink()
    monkeypatch.setitem(sys.modules, "cowork.channels.plugins.slack", None)
    store_setting(deployment, key="channel.slack.bot_token", value=Fernet(old).encrypt(SECRET_CHANNEL.encode()).decode())

    printed = _refused(deployment, capsys, names="setting channel.slack.bot_token (global)")

    assert "open with the legacy key or an older key in the key file, not with its first key" in printed
    assert SECRET_CHANNEL not in printed


# ── a run that stops part-way, and what stops one before it writes ─────────


def test_a_folder_at_a_temporary_name_stops_it_before_any_write(deployment, capsys):
    plain_records(deployment)
    (deployment["vaults"]["b"] / "postgres-orders.tmp").mkdir()

    _refused(deployment, capsys, names="postgres-orders.tmp: isn't a regular file")


def test_a_record_too_large_to_read_back_once_sealed_stops_it_before_any_write(deployment, capsys):
    """About 790 KB of plain text reads back plain, and seals to more than the 1 MiB the server
    reads, which SealedDataVault.save refuses: found before the marker, nothing is written."""
    plain_records(deployment)
    LocalDataVault(deployment["vaults"]["b"]).save("postgres", "big", {"host": "db.example", "sslrootcert": "A" * 790_000})
    assert (deployment["vaults"]["b"] / "postgres-big").stat().st_size < 1 << 20

    printed = _refused(deployment, capsys, names=f"{deployment['vaults']['b'] / 'postgres-big'}: sealed, its file would be ")

    assert "and the server reads at most 1048576" in printed
    assert "AAAA" not in printed


def test_a_plain_record_with_a_field_named_like_the_mark_is_sealed_as_plain(deployment):
    """Only the integer mark SealedDataVault.save writes makes a record sealed; a plain record's
    string "1" there is a field like any other, and the save writes the mark in its place."""
    plain_records(deployment)
    LocalDataVault(deployment["vaults"]["base"]).save("http", "flagged", {"token": "flag-token", SEALED_FIELD: "1"})

    assert run(deployment) == 0

    keys = MultiFernet([file_fernet(deployment)])
    assert SealedDataVault(deployment["vaults"]["base"], keys=keys).load("http", "flagged") == {"token": "flag-token"}
    on_disk = json.loads((deployment["vaults"]["base"] / "http-flagged").read_text())
    assert on_disk["fields"][SEALED_FIELD] == 1 and "flag-token" not in json.dumps(on_disk)


@pytest.mark.parametrize("secure_keys", [5, "password", ["password", 1]])
def test_a_record_whose_secure_keys_isnt_a_list_of_names_stops_it(deployment, capsys, secure_keys):
    plain_records(deployment)
    record_file = deployment["vaults"]["a"] / "postgres-sales"
    record = json.loads(record_file.read_text())
    record["secure_keys"] = secure_keys
    record_file.write_text(json.dumps(record))

    printed = _refused(deployment, capsys, names="postgres-sales: isn't a saved connection")

    assert SECRET_A not in printed


def test_a_vault_folder_another_user_owns_stops_it(deployment, monkeypatch, capsys):
    plain_records(deployment)
    owner = deployment["vaults"]["a"].stat().st_uid
    monkeypatch.setattr(os, "geteuid", lambda: owner + 1)

    printed = _refused(deployment, capsys, names=f"{deployment['vaults']['a']}: belongs to user {owner}, not to this command's user {owner + 1}")

    assert "Run it as the server's own user, with the server stopped." in printed


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX modes, enforced for non-root")
def test_a_vault_folder_this_user_cant_write_stops_it(deployment, capsys):
    plain_records(deployment)
    folder = deployment["vaults"]["b"]
    folder.chmod(0o500)
    try:
        _refused(deployment, capsys, names=f"{folder}: isn't writable by this command's user")
    finally:
        folder.chmod(0o700)


def test_a_run_an_error_stops_part_way_exits_2_and_the_same_run_again_finishes(deployment, monkeypatch, capsys):
    plain_records(deployment)
    store_setting(deployment, key="anthropic_api_key", value=legacy_fernet(deployment).encrypt(SECRET_SETTING.encode()).decode())
    save = SealedDataVault.save
    saved: list[str] = []

    def save_until_the_disk_fills(vault, engine, name, credentials, *, secure_keys=None):
        if saved:
            raise OSError(errno.ENOSPC, "No space left on device")
        saved.append(f"{engine}/{name}")
        return save(vault, engine, name, credentials, secure_keys=secure_keys)

    monkeypatch.setattr(SealedDataVault, "save", save_until_the_disk_fills)

    assert run(deployment) == 2

    printed = capsys.readouterr()
    assert printed.out == ""
    assert printed.err == (
        "error: The re-seal stopped part-way through its writes (No space left on device). Some values may already "
        "be sealed under the key file's first key, and the legacy key was kept. Fix the cause, then run the same "
        "command again with the same key file.\n"
    )
    assert deployment["legacy"].exists() and marker_present(deployment) and len(saved) == 1

    monkeypatch.setattr(SealedDataVault, "save", save)
    assert run(deployment) == 0

    keys = MultiFernet([file_fernet(deployment)])
    assert SealedDataVault(deployment["vaults"]["a"], keys=keys).load("postgres", "sales")["password"] == SECRET_A
    assert SealedDataVault(deployment["vaults"]["b"], keys=keys).load("postgres", "orders") == {"password": "b-secret"}
    assert SealedDataVault(deployment["vaults"]["base"], keys=keys).load("http", "status") == {"token": "base-token"}
    assert file_fernet(deployment).decrypt(setting_value(deployment, key="anthropic_api_key").encode()) == SECRET_SETTING.encode()
    assert not deployment["legacy"].exists()


# ── the other refusals ──────────────────────────────────────────────────────


def test_an_org_deployment_stops_it(deployment, org_deployment, capsys):
    plain_records(deployment)

    _refused(deployment, capsys, names="The re-seal command runs on a local deployment only.")


def test_a_sealed_record_no_key_opens_stops_it(deployment, capsys):
    plain_records(deployment)
    foreign = MultiFernet([Fernet(Fernet.generate_key())])
    SealedDataVault(deployment["vaults"]["a"], keys=foreign).save("postgres", "sales", {"password": SECRET_A})

    printed = _refused(deployment, capsys, names=f"connection postgres/sales in {deployment['vaults']['a']}: its password field doesn't open")

    assert SECRET_A not in printed


def test_a_malformed_legacy_key_stops_it(deployment, capsys):
    plain_records(deployment)
    deployment["legacy"].write_bytes(b"not-a-fernet-key")

    _refused(deployment, capsys, names=f"The old key in {deployment['legacy']} isn't a Fernet key.")


def test_a_managed_folder_without_its_vault_path_stops_it(deployment, capsys):
    plain_records(deployment)

    _refused(deployment, capsys, names="COWORK_MANAGED_DIR is set: pass --managed-vault", arguments=[])


# ── where the legacy key comes from ─────────────────────────────────────────


@pytest.mark.parametrize("source", ["environment", "env-file"])
def test_turning_on_with_the_legacy_key_in_cowork_master_key_names_where_it_is_left(deployment, monkeypatch, capsys, source):
    plain_records(deployment)
    key = Fernet.generate_key()
    if source == "environment":
        monkeypatch.setenv("COWORK_MASTER_KEY", key.decode())
        where = "the environment that starts Cowork"
    else:
        env_file = deployment["root"] / "home" / ".env"
        env_file.write_text(f"COWORK_MASTER_KEY={key.decode()}\n")
        monkeypatch.setitem(type(get_app_settings()).model_config, "env_file", [str(env_file)])
        where = f"the .env file that sets it ({env_file})"
    get_app_settings.cache_clear()
    store_setting(deployment, key="anthropic_api_key", value=Fernet(key).encrypt(SECRET_SETTING.encode()).decode())

    assert run(deployment) == 0

    assert file_fernet(deployment).decrypt(setting_value(deployment, key="anthropic_api_key").encode()) == SECRET_SETTING.encode()
    assert capsys.readouterr().out.splitlines()[1:] == [
        "The old key is still set as COWORK_MASTER_KEY, which the server ignores while COWORK_KEY_FILE is set: "
        f"remove it from {where}.",
        f"The old key file {deployment['legacy']} is still there too: delete it as well.",
    ]
    assert deployment["legacy"].exists()


def test_turning_on_with_an_old_key_file_seals_under_the_key_file_and_removes_it(deployment, capsys):
    plain_records(deployment)
    old_key_file = deployment["root"] / "old.key"
    deployment["legacy"].rename(old_key_file)
    store_setting(deployment, key="anthropic_api_key", value=Fernet(old_key_file.read_bytes()).encrypt(SECRET_SETTING.encode()).decode())

    assert run(deployment, "--old-key-file", str(old_key_file)) == 0

    assert file_fernet(deployment).decrypt(setting_value(deployment, key="anthropic_api_key").encode()) == SECRET_SETTING.encode()
    assert capsys.readouterr().out.splitlines()[-1] == f"Removed the old key file {old_key_file}."
    assert not old_key_file.exists()


def test_a_field_named_in_secure_keys_stays_masked_after_the_re_seal(deployment):
    """The name heuristic doesn't take client_id for a secret; the record's own secure_keys does."""
    LocalDataVault(deployment["vaults"]["base"]).save(
        "postgres", "crm", {"host": "db.example", "client_id": "zx-canary-7790"}, secure_keys=["client_id"]
    )

    assert run(deployment) == 0

    detail = ConnectionsService(LOCAL_SCOPE).get("postgres", "crm")
    assert detail is not None
    assert detail.fields == {"host": "db.example", "client_id": VAULT_KEEP_SENTINEL}
    assert detail.secure_keys == ["client_id"]
