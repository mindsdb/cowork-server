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
# One conversion process per server process: a hosted replica has 1 GiB for
# the server and its conversion together.
_child_slot = threading.BoundedSemaphore(1)

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


def child_command(*, limits: ChildLimits) -> tuple[list[str], dict[str, str]]:
    """The command and environment a conversion process starts with.

    ``-I`` keeps the server's environment, the user's site folder and the
    current directory out of the process, and ``-B`` keeps it from writing
    bytecode. It gets no environment of its own, so no credential. Windows
    needs ``SYSTEMROOT`` to start an interpreter and ``USERPROFILE`` to find
    the home folder ``cowork.common.paths`` reads at import; neither holds a
    secret. It runs the server's own interpreter,
    so it imports the same ``cowork``. On Windows a venv's ``python.exe``
    starts the base interpreter as its own child, which ``_kill`` and the
    watch reach too.
    """
    environment: dict[str, str] = {}
    if sys.platform == "win32":
        for name in ("SYSTEMROOT", "USERPROFILE"):
            environment[name] = os.environ[name]
    cpu_seconds = math.ceil(limits.wall_seconds) + 1
    command = [
        sys.executable, "-I", "-B", "-m", CHILD_MODULE, str(limits.memory_bytes), str(cpu_seconds)
    ]
    return command, environment


def convert_in_child(
    *, request: ConvertRequest, source: bytes, read_asset: AssetReader, limits: ChildLimits
) -> Conversion:
    """Convert *source* in a conversion process, serving its reads through *read_asset*.

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
    *, request: ConvertRequest, source: bytes, read_asset: AssetReader, limits: ChildLimits
) -> Conversion:
    command, environment = child_command(limits=limits)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        env=environment,
        creationflags=_CREATION_FLAGS,
    )
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
    return _outcome(exchange=exchange, returncode=process.returncode, cause=watch.cause)


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
    *, process: subprocess.Popen, request: ConvertRequest, source: bytes, read_asset: AssetReader
) -> _Exchange:
    """Send the request and answer reads until the process sends its result or stops."""
    served = 0
    try:
        write_frame(stream=process.stdin, message=request, payload=source)
        while True:
            received = read_frame(stream=process.stdout, adapter=CHILD_MESSAGE)
            if received is None:
                return _Exchange(message=None, payload=b"", broken=None)
            message, payload = received
            if not isinstance(message, AssetRequest):
                return _Exchange(message=message, payload=payload, broken=None)
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


def _serve(
    *, asked: AssetRequest, request: ConvertRequest, read_asset: AssetReader, served: int
) -> tuple[AssetReply, bytes]:
    """Answer one read. Every legitimate read sits below the source's folder.

    The inliner reads each asset at most once, after counting the reference,
    so a conversion asks at most ``max_references`` times.
    """
    base = request.base_parts
    if served >= request.limits.max_references:
        return AssetReply(refused=OmittedReason.BUDGET), b""
    if asked.parts[: len(base)] != base:
        return AssetReply(refused=OmittedReason.MISSING), b""
    try:
        content = read_asset(asked.parts)
    except ArtifactFileRefused as exc:
        return AssetReply(refused=exc.reason), b""
    return AssetReply(device=content.device, inode=content.inode), content.data


def _outcome(*, exchange: _Exchange, returncode: int, cause: str | None) -> Conversion:
    """The conversion, or the exception that answers for it.

    A complete result counts whatever ended the process afterwards.
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
        logger.warning("Export conversion stopped", extra={"cause": cause})
        raise ConversionOverrun(cause)
    if isinstance(message, ConversionFailed):
        logger.warning(
            "Export conversion failed",
            extra={"error_type": message.error_type, "detail": message.detail},
        )
        raise ConversionError(message.error_type)
    logger.warning(
        "Export conversion ended without a result",
        extra={"returncode": returncode, "protocol": exchange.broken},
    )
    raise ConversionError(f"the conversion process exited {returncode} without a result")


class _Watch:
    """Kills the conversion process past its wall clock or its memory limit.

    The wall clock counts from the start, imports included. Memory is what
    ``_held_bytes`` reads, checked every ``_WATCH_SECONDS``, so a fast
    allocation can pass the limit by what it takes in one check. On Linux the
    process's own address-space limit is the hard bound.
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
        try:
            watched = psutil.Process(self._process.pid)
        except psutil.Error:
            return
        while not self._stopped.wait(_WATCH_SECONDS):
            if self._process.poll() is not None:
                return
            if time.monotonic() >= self._deadline:
                self._stop(cause="wall clock")
                return
            try:
                held = _held_bytes(watched=watched)
            except psutil.Error:
                return
            if held > self._limits.memory_bytes:
                self._stop(cause="memory")
                return

    def _stop(self, *, cause: str) -> None:
        self.cause = cause
        _kill(self._process)
