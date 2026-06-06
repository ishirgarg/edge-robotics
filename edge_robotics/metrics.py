"""Turn raw measurements into the 5-metric row: Vision/VLM/Action (ms & %), E2E (ms), Freq (Hz).

How each number is derived:
  - E2E (ms)  : median of device-synced steady-state wall times of the full sample_actions.
  - Freq (Hz) : 1000 / E2E.
  - Vision / VLM / Action : GPU device time per phase, summed from the JAX profiler trace by
    `jax.named_scope` tag (see profiling/jax_profiler.py). This is a DIRECT measurement, not an
    inferred split.
  - %         : each phase as a share of (Vision + VLM + Action).
  - residual  : device time outside any phase scope (input transpose, noise sampling, the
    x_t+dt*v_t update, action_out_proj, D2D copies). Reported, not hidden.

If the trace is unavailable/unparseable the row degrades to E2E/Freq only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def stats(samples_ms: list[float]) -> dict:
    a = np.asarray(samples_ms, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


@dataclass
class PhaseRow:
    prompt_len: int
    num_steps: int
    e2e_ms: float
    freq_hz: float
    vision_ms: float | None
    vlm_ms: float | None
    action_ms: float | None
    vision_pct: float | None
    vlm_pct: float | None
    action_pct: float | None
    method: str
    notes: str = ""
    e2e_stats: dict = field(default_factory=dict)
    trace: dict | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_row(
    *,
    prompt_len: int,
    num_steps: int,
    e2e_samples: list[float],
    trace: dict | None,
    meta: dict,
) -> PhaseRow:
    est = stats(e2e_samples)
    e2e_ms = est["median"]
    freq_hz = 1000.0 / e2e_ms if e2e_ms > 0 else float("nan")

    vision_ms = vlm_ms = action_ms = None
    vision_pct = vlm_pct = action_pct = None
    notes_parts: list[str] = []

    if trace is not None and trace.get("ok"):
        p = trace["phases_ms_per_infer"]
        vision_ms, vlm_ms, action_ms = p["vision"], p["vlm"], p["action"]
        method = trace.get("method", "trace-namescope")

        total = vision_ms + vlm_ms + action_ms
        if total > 0:
            vision_pct = 100.0 * vision_ms / total
            vlm_pct = 100.0 * vlm_ms / total
            action_pct = 100.0 * action_ms / total

        # Trustworthiness signals: how much GPU time was attributable, and how the summed device
        # time compares to the measured wall E2E.
        frac = trace.get("attributed_frac")
        if frac is not None and frac < 0.90:
            notes_parts.append(f"only {frac*100:.1f}% of GPU time attributed (CUDA graphs on?)")
        residual = trace.get("residual_ms_per_infer", 0.0)
        gpu_total = trace.get("total_gpu_ms_per_infer", total + residual)
        notes_parts.append(f"residual(non-phase)={residual:.3f}ms")
        if e2e_ms > 0:
            disc = abs(gpu_total - e2e_ms) / e2e_ms * 100.0
            notes_parts.append(f"GPU_total={gpu_total:.3f}ms vs E2E={e2e_ms:.3f}ms ({disc:.1f}% diff)")
    else:
        method = "e2e-only"
        err = (trace or {}).get("error", "no trace")
        notes_parts.append(f"no usable trace ({err}): only E2E/Freq available")

    return PhaseRow(
        prompt_len=prompt_len,
        num_steps=num_steps,
        e2e_ms=e2e_ms,
        freq_hz=freq_hz,
        vision_ms=vision_ms,
        vlm_ms=vlm_ms,
        action_ms=action_ms,
        vision_pct=vision_pct,
        vlm_pct=vlm_pct,
        action_pct=action_pct,
        method=method,
        notes="; ".join(notes_parts),
        e2e_stats=est,
        trace=trace,
        meta=meta,
    )
