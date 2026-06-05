"""Device-synced steady-state wall timing.

This is NOT a naive `time.time()` measurement: every iteration ends with the system's
`block` (jax.block_until_ready), which forces the asynchronously-dispatched GPU work to
actually finish before the clock is read. That is the correct way to time async accelerator
work and is exactly what a profiler measures for a wall-clock span. Warmup iterations absorb
JIT compilation so only steady-state is reported.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def time_callable(
    fn: Callable[[], Any],
    block: Callable[[Any], Any],
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    """Run fn() warmup times (discarded), then iters times, returning per-iter milliseconds."""
    if iters <= 0:
        raise ValueError("iters must be >= 1")
    for _ in range(warmup):
        block(fn())
    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        block(out)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return samples_ms
