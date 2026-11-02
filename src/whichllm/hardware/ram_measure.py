"""Measured system RAM bandwidth, shown by ``whichllm hardware``.

Estimates keep the bandwidth read from the memory modules: the CPU speed
model is calibrated on that figure, and a measurement moves with whatever
else the machine is doing. The measurement shows how much of it the machine
reaches right now.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_MiB = 1024**2
# Bytes on each side of the copy, well past any desktop L3 cache.
_COPY_BYTES = 256 * _MiB
_MIN_SLICE_BYTES = 4 * _MiB
# Thread counts tried in turn. A desktop's two channels fill up near eight
# threads and lose to contention past that; a server needs dozens.
_THREAD_LADDER = (2, 4, 8, 16, 32, 64)
_COPIES_PER_PASS = 4
_PASSES = 3
_BARRIER_TIMEOUT_SECONDS = 30


def _usable_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _copy_rate(source: int, target: int, span: int, threads: int) -> float | None:
    """GB/s for ``threads`` threads each copying its own slice of ``span``."""
    size = span // threads
    barrier = threading.Barrier(threads + 1, timeout=_BARRIER_TIMEOUT_SECONDS)
    # Each worker times its own copies: this thread can wake from the barrier
    # well after the copies began, which would shorten the pass.
    starts = [0.0] * threads
    ends = [0.0] * threads

    def copy(index: int) -> None:
        offset = index * size
        try:
            for _ in range(_PASSES):
                barrier.wait()
                starts[index] = time.perf_counter()
                for _ in range(_COPIES_PER_PASS):
                    ctypes.memmove(target + offset, source + offset, size)
                ends[index] = time.perf_counter()
                barrier.wait()
        except threading.BrokenBarrierError:
            pass

    workers = [
        threading.Thread(target=copy, args=(index,), daemon=True)
        for index in range(threads)
    ]
    moved = 2 * size * threads * _COPIES_PER_PASS
    best = 0.0
    try:
        for worker in workers:
            worker.start()
        for _ in range(_PASSES):
            barrier.wait()
            barrier.wait()
            elapsed = max(ends) - min(starts)
            if elapsed > 0:
                best = max(best, moved / elapsed / 1e9)
    except (RuntimeError, threading.BrokenBarrierError) as error:
        barrier.abort()
        logger.debug(f"RAM bandwidth measurement failed: {error}")
        return None
    finally:
        for worker in workers:
            if worker.ident is not None:
                worker.join()
    return best or None


def measure_ram_bandwidth_gbps(available_bytes: int | None = None) -> float | None:
    """Copy bandwidth in GB/s, counting the bytes read and the bytes written.

    As in STREAM's Copy, threads copy their own slices at the same time
    (``ctypes.memmove`` releases the GIL). The thread count grows until it
    stops helping, and the fastest pass counts. Returns None when the
    buffers do not fit or the copy cannot run.
    """
    span = _COPY_BYTES
    if available_bytes is not None:
        # Source and target together stay under a quarter of the free RAM.
        span = min(span, available_bytes // 8)
    if span < _MIN_SLICE_BYTES:
        return None
    try:
        source = ctypes.create_string_buffer(span)
        target = ctypes.create_string_buffer(span)
    except MemoryError:
        return None
    # Pages never written all map to one zero page, which reads from cache.
    ctypes.memset(source, 0x5A, span)
    ctypes.memset(target, 0x5A, span)

    most = min(_usable_cpus(), span // _MIN_SLICE_BYTES)
    best = 0.0
    for threads in sorted({min(count, most) for count in _THREAD_LADDER}):
        rate = _copy_rate(
            ctypes.addressof(source), ctypes.addressof(target), span, threads
        )
        if rate is None:
            break
        if rate < best * 0.9:
            break
        best = max(best, rate)
    return best or None
