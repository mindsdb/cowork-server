"""Managed-vault startup failures identify a file without disclosing its record."""

import json
import logging
import os
from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault

import cowork.server as cowork_server


@pytest.mark.parametrize("log_format", ["json", "text"])
def test_open_denied_after_access_passes_logs_the_file_and_refuses_startup(
    managed_dir, monkeypatch, caplog, owned_logger, log_format
):
    vault = LocalDataVault(managed_dir.vault_dir)
    vault.save("postgres", "warehouse", {
        "password": "private-password-sentinel",
        "host": "private-record-field-sentinel",
    })
    filename = "postgres-warehouse"
    path = managed_dir.vault_dir / filename
    assert os.access(path, os.R_OK)
    real_open = os.open

    def deny_open(file, flags, *args, **kwargs):
        if Path(file).name == filename:
            raise PermissionError(13, "private-exception-sentinel", str(path))
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_open)
    monkeypatch.setenv("LOG_FORMAT", log_format)
    output = owned_logger(cowork_server.logger.name, level=logging.WARNING)
    with pytest.raises(RuntimeError, match="connection file .* can't be read"):
        cowork_server.create_app()

    [record] = [
        record for record in caplog.records
        if record.name == cowork_server.logger.name
        and getattr(record, "event", None) == "managed_connection_unreadable"
    ]
    assert record.levelno == logging.ERROR
    assert record.managed_connection_file == filename
    assert record.error_type == "PermissionError"
    assert record.exc_info is None
    rendered = output.stream.getvalue()
    assert filename in rendered
    assert "Cowork cannot read a managed connection" in rendered
    assert "private-" not in rendered
    assert str(managed_dir.root) not in rendered
    if log_format == "json":
        [diagnostic] = [json.loads(line) for line in rendered.splitlines()]
        assert diagnostic["managed_connection_file"] == filename
        assert diagnostic["reason"] == "file_unreadable"


def test_unreadable_vault_enumeration_logs_the_store_without_a_fake_filename(
    managed_dir, monkeypatch, caplog, owned_logger
):
    real_scandir = os.scandir
    vault_stat = managed_dir.vault_dir.stat()

    def deny_scandir(path):
        info = os.fstat(path) if isinstance(path, int) else Path(path).stat()
        if (info.st_dev, info.st_ino) == (vault_stat.st_dev, vault_stat.st_ino):
            raise PermissionError(13, "private-enumeration-sentinel", str(managed_dir.vault_dir))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", deny_scandir)
    output = owned_logger(cowork_server.logger.name, level=logging.WARNING)
    with pytest.raises(RuntimeError, match="data-vault.*can't be read"):
        cowork_server.create_app()
    [record] = [
        record for record in caplog.records
        if record.name == cowork_server.logger.name
        and getattr(record, "event", None) == "managed_connection_unreadable"
    ]
    assert record.reason == "vault_unreadable"
    assert record.error_type == "PermissionError"
    assert not hasattr(record, "managed_connection_file")
    assert "private-" not in output.stream.getvalue()
    assert str(managed_dir.root) not in output.stream.getvalue()


@pytest.mark.parametrize("has_connection", [False, True])
def test_readable_managed_vault_emits_no_unreadable_diagnostic(
    managed_dir, caplog, owned_logger, has_connection
):
    if has_connection:
        LocalDataVault(managed_dir.vault_dir).save("postgres", "warehouse", {"host": "db"})
    owned_logger(cowork_server.logger.name, level=logging.WARNING)
    cowork_server.create_app()
    assert not [
        record for record in caplog.records
        if record.name == cowork_server.logger.name
        and getattr(record, "event", None) == "managed_connection_unreadable"
    ]


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
def test_a_readable_managed_connection_link_keeps_its_existing_behavior(
    managed_dir, tmp_path, caplog, owned_logger
):
    source = LocalDataVault(tmp_path / "operator-vault")
    source.save("postgres", "warehouse", {"host": "db"})
    (managed_dir.vault_dir / "postgres-warehouse").symlink_to(
        tmp_path / "operator-vault" / "postgres-warehouse"
    )
    owned_logger(cowork_server.logger.name, level=logging.WARNING)
    cowork_server.create_app()
    assert not [
        record for record in caplog.records
        if record.name == cowork_server.logger.name
        and getattr(record, "event", None) == "managed_connection_unreadable"
    ]


def test_managed_connection_probe_closes_the_file_when_fstat_fails(
    managed_dir, monkeypatch, owned_logger
):
    LocalDataVault(managed_dir.vault_dir).save("postgres", "warehouse", {"host": "db"})
    real_open = os.open
    real_fstat = os.fstat
    opened = []

    def track_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path).name == "postgres-warehouse":
            opened.append(fd)
        return fd

    def deny_fstat(fd):
        if fd in opened:
            raise OSError("private-stat-sentinel")
        return real_fstat(fd)

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "fstat", deny_fstat)
    output = owned_logger(cowork_server.logger.name, level=logging.WARNING)
    with pytest.raises(RuntimeError, match="connection file .* can't be read"):
        cowork_server.create_app()
    [fd] = opened
    with pytest.raises(OSError):
        real_fstat(fd)
    assert "private-" not in output.stream.getvalue()
