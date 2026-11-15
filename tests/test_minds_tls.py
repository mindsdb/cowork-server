import pytest

from cowork.common.minds_tls import minds_ssl_verify


def test_verification_is_on_when_unset(monkeypatch):
    monkeypatch.delenv("ANTON_MINDS_SSL_VERIFY", raising=False)
    assert minds_ssl_verify() is True


@pytest.mark.parametrize("value", ["", "  ", "true", "TRUE", "1", "yes", "maybe"])
def test_verification_stays_on_unless_explicitly_off(monkeypatch, value):
    monkeypatch.setenv("ANTON_MINDS_SSL_VERIFY", value)
    assert minds_ssl_verify() is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", " OFF "])
def test_verification_turns_off_only_for_off_values(monkeypatch, value):
    monkeypatch.setenv("ANTON_MINDS_SSL_VERIFY", value)
    assert minds_ssl_verify() is False
