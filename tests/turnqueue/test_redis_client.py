from cowork.turnqueue.redis_client import get_redis, reset_redis


def test_get_redis_is_singleton(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_REDIS_URL", "redis://localhost:6379/0")
    reset_redis()
    a = get_redis()
    b = get_redis()
    assert a is b
