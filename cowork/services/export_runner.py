"""Run an export's conversion in a process the server kills past its limits.

One conversion process runs at a time in a server process, and a second
export answers busy at once. The server starts it
(``cowork.services.export_child``), sends it the source, answers each file it
asks for through the server's own pinned reader, and takes the result. A
thread beside it kills the process when it passes its wall clock or holds
more memory than its limit. A conversion that a limit stopped, or that
refused its source, is an overrun, which the export answers 413. Nothing the
process sends is trusted: every frame is bounded and validated, every read
goes through the reader with its own bounds, and every read it asks for must
sit below the source's folder.

The process runs as the server's user. The limits bound time, CPU and memory
against a hostile document; they don't confine a parser exploit.
"""
from __future__ import annotations

import logging
import math
import os
import site
import subprocess
import sys
import threading
import time
from typing import IO, NamedTuple

import psutil

from cowork.schemas.artifact_export import OmittedReason
from cowork.schemas.export_child import (
    AssetReply,
    AssetRequest,
    ChildLimits,
    ConversionFailed,
    ConversionRequest,
    Converted,
    ConvertRequest,
)
from cowork.services.artifact_files import ArtifactFileRefused
from cowork.services.artifact_inline import AssetReader
from cowork.services.export_child import (
    CHILD_MESSAGE,
    EXIT_REFUSED,
    Conversion,
    FrameError,
    read_frame,
    write_frame,
)

logger = logging.getLogger(__name__)

CHILD_MODULE = "cowork.services.export_child"
# How often the watch checks the process's clock and memory.
_WATCH_SECONDS = 0.1
# How long a process whose frames stopped short gets to exit on its own, so
# its exit, such as a signal at its CPU-time limit, explains the short frame.
_EXIT_GRACE_SECONDS = 1.0
# How much of the process's error output the server keeps for its log: the
# end of a traceback from a process that failed before it could report.
_ERROR_TAIL_BYTES = 2048
# One conversion process per server process: a hosted replica has 1 GiB for
# the server and its conversion together.
_child_slot = threading.BoundedSemaphore(1)
# The folder the process starts in: the filesystem root (on Windows, the
# current drive's), so it inherits no folder of the server's, and the server's
# user can't write there.
CHILD_CWD = os.path.abspath(os.sep)

if sys.platform == "win32":
    _CREATION_FLAGS = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
else:
    _CREATION_FLAGS = 0


class ConversionBusy(Exception):
    """Another conversion holds this server process's one slot."""


class ConversionOverrun(Exception):
    """The conversion passed a limit or refused its source. Nothing was converted."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


class ConversionError(Exception):
    """The conversion process failed or broke the protocol."""


class _Exchange(NamedTuple):
    """How the talk with the process ended: its last message, or why there was none."""

    message: Converted | ConversionFailed | None
    payload: bytes
    broken: str | None
    # The process still ran after it broke the protocol, so it was killed.
    killed_for_protocol: bool = False


class ChildCommand(NamedTuple):
    """The command line a conversion process starts with, and its whole environment."""

    argv: list[str]
    environment: dict[str, str]


class ServedRead(NamedTuple):
    """The server's answer to one read, and the bytes it carries."""

    reply: AssetReply
    data: bytes


def child_command(*, limits: ChildLimits) -> ChildCommand:
    """The command and environment a conversion process starts with.

    ``-I`` keeps the server's environment, the user's site folder and the
    current directory out of the process, and ``-B`` keeps it from writing
    bytecode. A server whose interpreter uses the user's site folder, such as
    one installed with ``pip install --user``, may import ``cowork`` from
    there, so its process keeps that folder (``-E -P`` in place of ``-I``)
    and gets ``PYTHONUSERBASE`` set to the server's own user base, a custom
    one included.

    Otherwise it gets only the variables that find the home folder, which
    ``cowork.common.paths`` reads at import: ``HOME``, and on Windows
    ``USERPROFILE``, plus ``SYSTEMROOT`` to start an interpreter. None holds
    a secret.
    It runs the server's own interpreter, so it imports the same ``cowork``.
    On Windows a venv's ``python.exe`` starts the base interpreter as its own
    child, which ``_kill`` and the watch reach too.
    """
    user_site = bool(site.ENABLE_USER_SITE)
    names = ["HOME"] + (["SYSTEMROOT", "USERPROFILE"] if sys.platform == "win32" else [])
    environment = {name: os.environ[name] for name in names if name in os.environ}
    if user_site:
        environment["PYTHONUSERBASE"] = site.getuserbase()
    isolation = ["-E", "-P"] if user_site else ["-I"]
    cpu_seconds = math.ceil(limits.wall_seconds) + 1
    argv = [
        sys.executable,
        *isolation,
        "-B",
        "-m",
        CHILD_MODULE,
        str(limits.memory_bytes),
        str(cpu_seconds),
    ]
    return ChildCommand(argv=argv, environment=environment)


def convert_in_child(
    *,
    request: ConversionRequest,
    source: bytes,
    read_asset: AssetReader | None = None,
    limits: ChildLimits,
) -> Conversion:
    """Convert *source* in a conversion process, serving its reads through *read_asset*.

    A ``PublishMarkdownRequest`` reads nothing and takes no *read_asset*, so
    a read it asks for breaks the protocol.

    Blocking: it runs in the export's worker thread, which holds the pinned
    folder *read_asset* reads. Raises ``ConversionBusy`` while another
    conversion runs, ``ConversionOverrun`` for one a limit stopped or that
    refused its source, and ``ConversionError`` for one that failed.
    """
    if not _child_slot.acquire(blocking=False):
        raise ConversionBusy("a conversion is already running")
    try:
        return _run(request=request, source=source, read_asset=read_asset, limits=limits)
    finally:
        _child_slot.release()


def _run(
    *,
    request: ConversionRequest,
    source: bytes,
    read_asset: AssetReader | None,
    limits: ChildLimits,
) -> Conversion:
    command = child_command(limits=limits)
    process = subprocess.Popen(
        command.argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        cwd=CHILD_CWD,
        env=command.environment,
        creationflags=_CREATION_FLAGS,
    )
    errors = _ErrorTail(stream=process.stderr)
    errors.start()
    watch = _Watch(process=process, limits=limits)
    watch.start()
    try:
        exchange = _talk(process=process, request=request, source=source, read_asset=read_asset)
        if exchange.broken is not None:
            try:
                process.wait(timeout=_EXIT_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _kill(process)
                exchange = exchange._replace(killed_for_protocol=True)
    except BaseException:
        _kill(process)
        raise
    finally:
        _close_quietly(stream=process.stdin)
        # The watch kills a process that doesn't exit by its wall clock.
        process.wait()
        watch.stop()
        _close_quietly(stream=process.stdout)
        error_tail = errors.finish()
    return _outcome(
        exchange=exchange, returncode=process.returncode, cause=watch.cause, error_tail=error_tail
    )


def _kill(process: subprocess.Popen) -> None:
    """Kill the conversion process, and on Windows the interpreter a venv's launcher started.

    On Linux and macOS the process can start no other: its process limit is 0.
    """
    if sys.platform == "win32":
        try:
            for child in psutil.Process(process.pid).children(recursive=True):
                child.kill()
        except psutil.Error:
            pass
    try:
        process.kill()
    except OSError:
        pass


def _held_bytes(*, watched: psutil.Process) -> int:
    """The memory the conversion holds: its resident set, or on Windows its private bytes.

    On Windows a venv's ``python.exe`` can be a launcher whose child does the
    work, so the launcher and every process under it count together.
    """
    if sys.platform != "win32":
        return watched.memory_info().rss
    held = watched.memory_info().private
    for child in watched.children(recursive=True):
        try:
            held += child.memory_info().private
        except psutil.Error:
            pass
    return held


def _close_quietly(*, stream: IO[bytes]) -> None:
    """Close one end of a pipe whose other end may be gone."""
    try:
        stream.close()
    except OSError:
        pass


def _talk(
    *,
    process: subprocess.Popen,
    request: ConversionRequest,
    source: bytes,
    read_asset: AssetReader | None,
) -> _Exchange:
    """Send the request and answer reads until the process sends its result or stops."""
    served = 0
    try:
        try:
            write_frame(stream=process.stdin, message=request, payload=source)
        except OSError:
            return _report_before_reading(process=process)
        while True:
            received = read_frame(stream=process.stdout, adapter=CHILD_MESSAGE)
            if received is None:
                return _Exchange(message=None, payload=b"", broken=None)
            message, payload = received
            if not isinstance(message, AssetRequest):
                return _Exchange(message=message, payload=payload, broken=None)
            if not isinstance(request, ConvertRequest) or read_asset is None:
                return _Exchange(message=None, payload=b"", broken="a read this request never makes")
            reply, data = _serve(
                asked=message, request=request, read_asset=read_asset, served=served
            )
            served += 1
            write_frame(stream=process.stdin, message=reply, payload=data)
    except FrameError as exc:
        return _Exchange(message=None, payload=b"", broken=str(exc))
    except OSError:
        # The process closed its end, killed or done: its exit decides.
        return _Exchange(message=None, payload=b"", broken=None)


def _report_before_reading(*, process: subprocess.Popen) -> _Exchange:
    """The report of a process that closed its input before it read the request.

    One that failed while it set its limits reports and exits without reading,
    and a request larger than the pipe then fails to write. Its exit closes
    its output, and the watch bounds the read, so this can't hang.
    """
    try:
        received = read_frame(stream=process.stdout, adapter=CHILD_MESSAGE)
    except (FrameError, OSError):
        received = None
    if received is not None and isinstance(received.message, ConversionFailed):
        return _Exchange(message=received.message, payload=b"", broken=None)
    return _Exchange(message=None, payload=b"", broken=None)


def _serve(
    *, asked: AssetRequest, request: ConvertRequest, read_asset: AssetReader, served: int
) -> ServedRead:
    """Answer one read. Every legitimate read sits below the source's folder.

    The inliner reads each asset at most once, after counting the reference,
    so a conversion asks at most ``max_references`` times.
    """
    base = request.base_parts
    if served >= request.limits.max_references:
        return ServedRead(reply=AssetReply(refused=OmittedReason.BUDGET), data=b"")
    if asked.parts[: len(base)] != base:
        return ServedRead(reply=AssetReply(refused=OmittedReason.MISSING), data=b"")
    try:
        content = read_asset(asked.parts)
    except ArtifactFileRefused as exc:
        return ServedRead(reply=AssetReply(refused=exc.reason), data=b"")
    return ServedRead(
        reply=AssetReply(device=content.device, inode=content.inode), data=content.data
    )


def _outcome(
    *, exchange: _Exchange, returncode: int, cause: str | None, error_tail: str
) -> Conversion:
    """The conversion, or the exception that answers for it.

    A complete result counts whatever ended the process afterwards. The log
    lines use the fields the server's log formatter keeps (``reason``,
    ``error_type``, ``stack``); a process that failed before it could report
    leaves the end of its error output in ``stack``, and so does one that
    refused its source, which writes there why.
    """
    message = exchange.message
    if isinstance(message, Converted):
        if message.denied_events:
            raise ConversionError("the conversion tried to connect or to start a process")
        return Conversion(
            data=exchange.payload,
            omitted_count=message.omitted_count,
            scripts_dropped=message.scripts_dropped,
        )
    if cause is None and not exchange.killed_for_protocol:
        if returncode == EXIT_REFUSED:
            cause = "refused"
        elif returncode < 0:
            # A signal: SIGXCPU or SIGKILL at the CPU-time limit, or a crash.
            cause = f"signal {-returncode}"
    if cause is not None:
        extra = {"event": "export_conversion_stopped", "reason": cause}
        if cause == "refused":
            # The refusal's types and locations, never its values.
            extra["stack"] = error_tail
        logger.warning("Export conversion stopped", extra=extra)
        raise ConversionOverrun(cause)
    if isinstance(message, ConversionFailed):
        logger.warning(
            "Export conversion failed",
            extra={
                "event": "export_conversion_failed",
                "error_type": message.error_type,
                "stack": message.detail,
            },
        )
        raise ConversionError(message.error_type)
    ended = f"exited {returncode}"
    if exchange.broken is not None:
        ended = f"{ended} after {exchange.broken}"
    logger.warning(
        "Export conversion ended without a result",
        extra={"event": "export_conversion_no_result", "reason": ended, "stack": error_tail},
    )
    raise ConversionError(f"the conversion process exited {returncode} without a result")


class _ErrorTail:
    """Drains the process's error output in a thread of its own and keeps its end.

    Drained, a flood of error output can't block the process; kept, the end
    of a traceback from a process that failed before it could report, such as
    an import that failed, reaches the log.
    """

    def __init__(self, *, stream: IO[bytes]) -> None:
        self._stream = stream
        self._tail = b""
        self._thread = threading.Thread(
            target=self._run, name="export-conversion-errors", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> str:
        """The kept end, once the process has exited and its error output closed."""
        self._thread.join(timeout=_EXIT_GRACE_SECONDS)
        _close_quietly(stream=self._stream)
        return self._tail.decode("utf-8", errors="replace")

    def _run(self) -> None:
        try:
            while chunk := self._stream.read1(64 * 1024):
                self._tail = (self._tail + chunk)[-_ERROR_TAIL_BYTES:]
        except (OSError, ValueError):
            # Closed under the read: the process is gone.
            return


class _Watch:
    """Kills the conversion process past its wall clock or its memory limit.

    The wall clock counts from the start, imports included. Memory is what
    ``_held_bytes`` reads, checked every ``_WATCH_SECONDS``, so a fast
    allocation can pass the limit by what it takes in one check. On Linux the
    process's own address-space limit is the hard bound. On macOS and Windows
    this watch is the only memory limit, so a running process whose memory
    can't be read is killed, not left to run unwatched.
    """

    def __init__(self, *, process: subprocess.Popen, limits: ChildLimits) -> None:
        self.cause: str | None = None
        self._process = process
        self._limits = limits
        self._deadline = time.monotonic() + limits.wall_seconds
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="export-conversion-watch")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join()

    def _run(self) -> None:
        watched: psutil.Process | None = None
        while not self._stopped.wait(_WATCH_SECONDS):
            if self._process.poll() is not None:
                return
            if time.monotonic() >= self._deadline:
                self._stop(cause="wall clock")
                return
            try:
                if watched is None:
                    watched = psutil.Process(self._process.pid)
                held = _held_bytes(watched=watched)
            except psutil.NoSuchProcess:
                # It exited under the read (on macOS an unreaped one raises
                # ZombieProcess), so its own exit decides the outcome. poll()
                # can't tell here: while _run waits on the process, it holds
                # the lock poll() needs and poll() answers None.
                return
            except psutil.Error:
                # A running process is never left without its limits.
                self._stop(cause="memory unreadable")
                return
            if held > self._limits.memory_bytes:
                self._stop(cause="memory")
                return

    def _stop(self, *, cause: str) -> None:
        self.cause = cause
        _kill(self._process)
