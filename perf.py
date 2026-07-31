"""
perf.py — Lightweight stage-timing helper for pipeline profiling.

Usage:
    from perf import timed

    with timed("asr.transcribe"):
        text = _asr.transcribe(audio_np)

Every call logs "<label>  <elapsed> ms" to the "perf" logger. api.py wires
that logger (via the root config) to both the console and logs/app.log, so
`grep perf logs/app.log` shows exactly where time went in a request.
"""

import logging
import time
from contextlib import contextmanager

log = logging.getLogger("perf")


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("%-28s %8.1f ms", label, elapsed_ms)
