"""The server starts the doc publish worker with the doc relay and stops it
after the turn drains, alongside the scratchpad closes."""
import inspect

from cowork import server


def test_the_worker_starts_with_the_relay_and_stops_alongside_the_scratchpad_closes():
    source = inspect.getsource(server.lifespan)
    start = source.index("start_doc_publish_worker()")
    assert source.index("start_doc_relay()") < start
    stop = source.index("stop_doc_publish_worker(")
    # After every turn drain, and in the same gather as the scratchpad closes:
    # the pod's grace period has no room for a wait of its own.
    assert source.index("drain_push_notifications()") < stop
    gather = source.rindex("asyncio.gather(", 0, stop)
    block = source[gather:source.index("\n        )", stop)]
    assert "drain_scratchpad_closes()" in block and "stop_doc_publish_worker(" in block
    # A failing worker stop must not cut the rest of the teardown short.
    assert "return_exceptions=True" in block


def test_the_worker_stop_fits_in_the_scratchpad_closes_wait():
    from cowork.common.chat_session import drain_scratchpad_closes
    from cowork.docs.publish_queue import STOP_BUDGET_S

    timeout = inspect.signature(drain_scratchpad_closes).parameters["timeout"].default
    assert STOP_BUDGET_S <= timeout
