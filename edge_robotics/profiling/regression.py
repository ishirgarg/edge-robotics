"""num_steps regression: isolate the Action (denoise) phase from Vision+VLM (encode+prefill)
using ONLY the public `sample_actions(num_steps=k)` API — no model internals touched.

pi-0.5 inference = [encode images] + [LLM prefill] + num_steps x [one denoise step].
Only the denoise loop repeats per step, so total device-synced time(k) is linear in k:

    time(k) = intercept + slope * k

  slope     = cost of one denoise step  ->  Action(N) = slope * N
  intercept = encode + prefill          ->  Vision + VLM (the once-per-inference part)

This gives a robust, parser-free phase split that also validates the JAX-profiler trace.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .walltime import time_callable


def fit_num_steps_regression(
    infer_at: Callable[[int], Callable[[], Any]],
    block: Callable[[Any], Any],
    *,
    steps_list: list[int],
    warmup: int,
    iters: int,
) -> dict:
    """Time inference at several denoise-step counts and fit a line. Returns slope/intercept (ms)."""
    if len(steps_list) < 2:
        raise ValueError("regression needs >= 2 distinct num_steps values")

    per_k: dict[int, dict] = {}
    ks: list[float] = []
    ys: list[float] = []
    for k in steps_list:
        samples = time_callable(infer_at(k), block, warmup=warmup, iters=iters)
        med = float(np.median(samples))
        per_k[k] = {"median_ms": med, "samples_ms": samples}
        ks.append(float(k))
        ys.append(med)

    ks_a = np.asarray(ks)
    ys_a = np.asarray(ys)
    # Least-squares fit y = intercept + slope * k.
    A = np.vstack([np.ones_like(ks_a), ks_a]).T
    (intercept, slope), *_ = np.linalg.lstsq(A, ys_a, rcond=None)
    pred = intercept + slope * ks_a
    ss_res = float(np.sum((ys_a - pred) ** 2))
    ss_tot = float(np.sum((ys_a - ys_a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "slope_ms_per_step": float(slope),
        "intercept_ms": float(intercept),
        "r2": float(r2),
        "steps_list": list(steps_list),
        "per_k": per_k,
    }
