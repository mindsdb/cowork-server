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
from collections.abc import Callable
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
from cowork.services.connectors.sealed_vault import SEALED_FIELD, SealedDataVault, is_fernet_token
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
    assert UNOPENED_ROW_REMEDY in printed and UNOPENED_CONNECTION_REMEDY not in printed


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


PARTLY_SEALED_HEADING = (
    "These saved connections hold sealed values but don't read as sealed records, so the re-seal wrote nothing:"
)
PARTLY_SEALED_REASON = "some of its values are sealed, but it doesn't read as a sealed record"
PARTLY_SEALED_REMEDY = "Remove each one and run this again; then connect it again once Cowork starts."
RESERVED_HEADING = "These saved connections hold a field named _sealed and aren't sealed, so the re-seal wrote nothing:"
RESERVED_REMEDY = (
    "Save each one again without that field, with Cowork started as before, or remove it; then run this again. "
    "A connection in the managed folder can't be saved from Cowork: edit its file to drop the field, or remove it."
)


def _partly_sealed(deployment, *, changed: dict[str, object]) -> Path:
    """A sealed record, its host and password both tokens, with ``changed`` written over its
    fields. Returns its file."""
    keys = MultiFernet([file_fernet(deployment)])
    SealedDataVault(deployment["vaults"]["a"], keys=keys).save("postgres", "mixed", {"host": "db.example", "password": SECRET_A})
    record_file = deployment["vaults"]["a"] / "postgres-mixed"
    record = json.loads(record_file.read_text())
    record["fields"].update(changed)
    record_file.write_text(json.dumps(record))
    return record_file


@pytest.mark.parametrize("mark", ["1", 1, True, 1.0], ids=["string", "int", "bool", "float"])
def test_a_plain_record_with_a_field_named_like_the_mark_stops_it(deployment, capsys, mark):
    """The field name _sealed is reserved for the sealed mark. A plain record that holds it,
    whatever the field holds, the integer 1 too, is neither sealed nor plain: sealed as plain, it
    would lose the field to the mark, and opened as sealed, no key would open its plain values."""
    plain_records(deployment)
    record_file = deployment["vaults"]["base"] / "acme_api-acme"
    LocalDataVault(deployment["vaults"]["base"]).save("acme_api", "acme", {"api_token": "flag-token", SEALED_FIELD: mark})

    printed = _refused(deployment, capsys, names=f"{record_file}: the field name _sealed is reserved for the sealed mark")

    assert RESERVED_HEADING in printed and RESERVED_REMEDY in printed
    assert PARTLY_SEALED_HEADING not in printed
    assert "flag-token" not in printed


@pytest.mark.parametrize("plain_value", ["db.example", 5432], ids=["text", "number"])
def test_a_record_with_the_mark_beside_values_that_arent_sealed_stops_it(deployment, capsys, plain_value):
    """A sealed record with one value put back as plain text or a number. Its refusal says to
    remove it, never to save it again. An edit from the form would keep the stored password, a
    token, as plain text, and the next run would seal that text as the password. With a key file,
    Cowork doesn't start beside it either."""
    plain_records(deployment)
    record_file = _partly_sealed(deployment, changed={"host": plain_value})

    printed = _refused(deployment, capsys, names=f"{record_file}: {PARTLY_SEALED_REASON}")

    assert PARTLY_SEALED_HEADING in printed and PARTLY_SEALED_REMEDY in printed
    assert RESERVED_HEADING not in printed and "Save each one again" not in printed
    assert SECRET_A not in printed and "db.example" not in printed


def test_a_record_with_a_string_mark_beside_tokens_stops_it_with_the_remove_remedy(deployment, capsys):
    """A sealed record whose mark was made the string "1": no plain value, and no mark the server
    reads as sealed. Saved again from the form, as the reserved field's remedy says, the edit would
    keep each token's text as a plain value, the next run would seal that text, and the password
    would open as a token's text. So its refusal says to remove it."""
    plain_records(deployment)
    record_file = _partly_sealed(deployment, changed={SEALED_FIELD: "1"})

    printed = _refused(deployment, capsys, names=f"{record_file}: {PARTLY_SEALED_REASON}")

    assert PARTLY_SEALED_HEADING in printed and PARTLY_SEALED_REMEDY in printed
    assert RESERVED_HEADING not in printed and "Save each one again" not in printed
    assert SECRET_A not in printed and "db.example" not in printed


def test_a_partly_sealed_record_and_a_plain_one_with_the_field_are_refused_under_their_own_headings(deployment, capsys):
    """Each kind is refused as itself: the partly sealed record first, and once it is removed, as its
    refusal says, the plain record with a field named like the mark under the reserved heading."""
    plain_records(deployment)
    partly_sealed = _partly_sealed(deployment, changed={"host": "db.example"})
    reserved = deployment["vaults"]["base"] / "acme_api-acme"
    LocalDataVault(deployment["vaults"]["base"]).save("acme_api", "acme", {"api_token": "flag-token", SEALED_FIELD: 1})

    first = _refused(deployment, capsys, names=f"{partly_sealed}: {PARTLY_SEALED_REASON}")

    assert PARTLY_SEALED_HEADING in first and str(reserved) not in first and RESERVED_HEADING not in first

    partly_sealed.unlink()
    second = _refused(deployment, capsys, names=f"{reserved}: the field name _sealed is reserved for the sealed mark")

    assert RESERVED_HEADING in second and RESERVED_REMEDY in second
    assert PARTLY_SEALED_HEADING not in second and str(partly_sealed) not in second


def _with_a_skipped_character(token: str, character: str = ".") -> str:
    """``token`` with ``character`` after its second character, inside the text every token starts
    with. Fernet's base64 decoder skips the character, so the token opens all the same."""
    return token[:2] + character + token[2:]


# Tokens Fernet opens whose shape is no token's: with stray whitespace a hand edit can leave around
# them, and with a character Fernet's decoder skips inside the text every token starts with. Only
# opening the value reads the last two as sealed.
OPENING_REWRITES = pytest.mark.parametrize(
    "rewrite",
    [
        lambda token: token + "\n",
        lambda token: " " + token,
        _with_a_skipped_character,
        lambda token: _with_a_skipped_character(token, '"'),
    ],
    ids=["trailing-newline", "leading-space", "dot-inside-the-prefix", "quote-inside-the-prefix"],
)


def _rewritten_password(
    deployment,
    *,
    fields: dict[str, object],
    rewrite: Callable[[str], str],
    mark: object,
    sealed_under: Fernet | None = None,
) -> Path:
    """A record SealedDataVault saved with ``fields`` under ``sealed_under``, the key file's key
    when None, its password token written back as ``rewrite`` makes it and its mark as ``mark``.
    Returns its file."""
    keys = MultiFernet([sealed_under or file_fernet(deployment)])
    SealedDataVault(deployment["vaults"]["a"], keys=keys).save("postgres", "padded", fields)
    record_file = deployment["vaults"]["a"] / "postgres-padded"
    record = json.loads(record_file.read_text())
    record["fields"]["password"] = rewrite(record["fields"]["password"])
    record["fields"][SEALED_FIELD] = mark
    record_file.write_text(json.dumps(record))
    return record_file


@OPENING_REWRITES
@pytest.mark.parametrize(
    "fields",
    [{"password": SECRET_A}, {"host": "db.example", "password": SECRET_A}],
    ids=["alone", "beside-a-good-token"],
)
@pytest.mark.parametrize("sealed_under", [file_fernet, legacy_fernet], ids=["key-file", "legacy-key"])
def test_a_token_that_opens_is_sealed_again_whatever_its_shape(deployment, rewrite, fields, sealed_under):
    """The run counts a value its keys open as sealed, whatever its shape, so a sealed record with
    one token a hand edit changed opens to its secret and is sealed again, rather than refused as a
    record with a field named like the mark. It reads each record with every key it opens with, the
    legacy key among them, not with the key file's keys alone, which it seals under: a token with a
    skipped character inside its first six characters reads as sealed only because it opens."""
    plain_records(deployment)
    record_file = _rewritten_password(
        deployment, fields=fields, rewrite=rewrite, mark=1, sealed_under=sealed_under(deployment)
    )

    assert run(deployment) == 0

    keys = MultiFernet([file_fernet(deployment)])
    assert SealedDataVault(deployment["vaults"]["a"], keys=keys).load("postgres", "padded") == fields
    on_disk = json.loads(record_file.read_text())["fields"]
    assert is_fernet_token(value=on_disk["password"]) and SECRET_A not in json.dumps(on_disk)


@OPENING_REWRITES
def test_a_string_mark_beside_a_changed_token_that_opens_stops_it_with_the_remove_remedy(deployment, capsys, rewrite):
    """The changed token opens with the run's keys, so the record holds a sealed value beside a mark
    the server doesn't read as sealed. Saved again from the form, as the reserved field's remedy
    says, the edit would keep the token's text as the password, so its refusal says to remove it."""
    plain_records(deployment)
    record_file = _rewritten_password(deployment, fields={"password": SECRET_A}, rewrite=rewrite, mark="1")

    printed = _refused(deployment, capsys, names=f"{record_file}: {PARTLY_SEALED_REASON}")

    assert PARTLY_SEALED_HEADING in printed and PARTLY_SEALED_REMEDY in printed
    assert RESERVED_HEADING not in printed and "Save each one again" not in printed
    assert SECRET_A not in printed


UNOPENED_HEADING = "No key opens these, so the re-seal changed nothing:"
UNOPENED_CONNECTION_REMEDY = (
    "Remove each connection no key you hold opens and run this again; then connect it again once Cowork starts."
)
UNOPENED_ROW_REMEDY = (
    "A row no key you hold opens can be entered again, or deleted, with Cowork started as before; then run this again."
)


def another_fernet(deployment) -> Fernet:
    """A key the deployment doesn't hold."""
    return Fernet(Fernet.generate_key())


@pytest.mark.parametrize(
    ("rewrite", "sealed_under"),
    [
        (lambda token, fernet: token[:-10], file_fernet),
        (lambda token, fernet: token + "\n", another_fernet),
        (lambda token, fernet: token[:3] + "\n " + token[3:], another_fernet),
        (lambda token, fernet: '"' + token + '"', another_fernet),
        (lambda token, fernet: " " + fernet.encrypt(b"not json").decode(), file_fernet),
        (lambda token, fernet: _with_a_skipped_character(fernet.encrypt(b"not json").decode()), file_fernet),
    ],
    ids=[
        "truncated", "another-key-with-a-trailing-newline", "another-key-with-whitespace-inside", "another-key-quoted",
        "opens-to-text-that-isnt-json", "opens-to-text-that-isnt-json-with-a-skipped-character",
    ],
)
def test_ciphertext_no_key_opens_under_the_mark_stops_it_as_unopened(deployment, capsys, rewrite, sealed_under):
    """A token cut short, one sealed under a key the run doesn't hold with whitespace a hand edit left
    after or inside it or quotes around it, and one Fernet opens to text that isn't the JSON a save
    seals: none opens to a value, and none has a token's shape. The run reads each one as sealed, as
    its text, with each character outside a token's alphabet removed, starts as a token's does, or as Fernet opens it, and refuses it as a value no key opens,
    by name, whose remedy is to remove the connection and connect it again. Read as plain, it would
    get the reserved field's remedy to save it again from the form, which keeps the ciphertext's text
    as the password."""
    plain_records(deployment)
    fernet = sealed_under(deployment)
    record_file = _rewritten_password(
        deployment,
        fields={"password": SECRET_A},
        rewrite=lambda token: rewrite(token, fernet),
        mark=1,
        sealed_under=fernet,
    )

    printed = _refused(
        deployment,
        capsys,
        names=f"connection postgres/padded in {record_file.parent}: its password field doesn't open with the server's key",
    )

    assert UNOPENED_HEADING in printed and UNOPENED_CONNECTION_REMEDY in printed
    assert RESERVED_HEADING not in printed and PARTLY_SEALED_HEADING not in printed
    assert "Save each one again" not in printed and "Traceback" not in printed
    assert SECRET_A not in printed


@pytest.mark.parametrize(
    "rewrite",
    [
        lambda token: token[:-10],
        lambda token: Fernet(Fernet.generate_key()).encrypt(b'"zx-other"').decode(),
        lambda token: "gAAAAA-plain-token",
    ],
    ids=["truncated", "another-key", "plain-text-with-the-prefix"],
)
def test_a_connection_no_key_opens_beside_a_good_token_is_removed_and_connected_again(deployment, capsys, rewrite):
    """A sealed record whose password no key opens, beside a host token that opens. Its refusal says
    to remove the connection and connect it again, never to enter it again: the form shows the host
    as its token's text and keeps that text, as it keeps each value the person doesn't type again
    (cowork.services.connectors.identity.resolve_keep_sentinels), and the next run would seal the
    token's text as the host. Followed, the remedy leaves each value opening to what was typed."""
    plain_records(deployment)
    fields = {"host": "db.example", "password": SECRET_A}
    record_file = _rewritten_password(deployment, fields=fields, rewrite=rewrite, mark=1)

    printed = _refused(
        deployment,
        capsys,
        names=f"connection postgres/padded in {record_file.parent}: its password field doesn't open with the server's key",
    )

    assert UNOPENED_HEADING in printed and UNOPENED_CONNECTION_REMEDY in printed
    assert "entered again" not in printed and UNOPENED_ROW_REMEDY not in printed

    record_file.unlink()
    assert run(deployment) == 0
    vault = SealedDataVault(deployment["vaults"]["a"], keys=MultiFernet([file_fernet(deployment)]))
    vault.save("postgres", "padded", fields)
    opened = vault.load("postgres", "padded")
    assert opened == fields and not any(str(value).startswith("gAAAAA") for value in opened.values())


def test_a_connection_and_a_setting_no_key_opens_each_get_their_remedy(deployment, capsys):
    plain_records(deployment)
    record_file = _rewritten_password(deployment, fields={"password": SECRET_A}, rewrite=lambda token: token[:-10], mark=1)
    store_setting(deployment, key="anthropic_api_key", value=Fernet(Fernet.generate_key()).encrypt(b"lost").decode())

    printed = _refused(deployment, capsys, names=f"connection postgres/padded in {record_file.parent}")

    assert "setting anthropic_api_key (global)" in printed
    assert UNOPENED_CONNECTION_REMEDY in printed and UNOPENED_ROW_REMEDY in printed


def test_a_plain_value_that_starts_as_a_token_does_beside_the_mark_stops_it_as_unopened(deployment, capsys):
    """A plain record with the integer mark beside a value whose text merely starts as a token's
    does reads as sealed, and is refused as a value no key opens, not as a field named like the
    mark. That false reading is accepted: its remedy removes the connection and connects it again,
    which costs a re-entry, and no real secret is sealed twice."""
    plain_records(deployment)
    base = deployment["vaults"]["base"]
    LocalDataVault(base).save("acme_api", "acme", {"api_token": "gAAAAA-plain-token", SEALED_FIELD: 1})

    printed = _refused(
        deployment,
        capsys,
        names=f"connection acme_api/acme in {base}: its api_token field doesn't open with the server's key",
    )

    assert UNOPENED_HEADING in printed and UNOPENED_CONNECTION_REMEDY in printed and RESERVED_HEADING not in printed
    assert "gAAAAA-plain-token" not in printed


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
