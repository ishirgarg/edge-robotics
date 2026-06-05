"""Turn raw measurements into the 5-metric row: Vision/VLM/Action (ms & %), E2E (ms), Freq (Hz).

How each number is derived (see plan §D):
  - E2E (ms)  : median of device-synced steady-state wall times of the full sample_actions.
  - Freq (Hz) : 1000 / E2E.
  - Action    : num_steps regression slope x num_steps  (robust, parser-free).
  - Vision+VLM: num_steps regression intercept (encode + prefill, once per inference).
  - Vision/VLM split: the JAX-profiler trace's vision:vlm device-time ratio applied to the
                      Vision+VLM total. If no trace, Vision/VLM are reported combined.
  - %         : each phase as a share of (Vision + VLM + Action).
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
    action_ms: float | None
    vision_vlm_ms: float | None
    vision_ms: float | None
    vlm_ms: float | None
    action_pct: float | None
    vision_pct: float | None
    vlm_pct: float | None
    method: str
    notes: str = ""
    e2e_stats: dict = field(default_factory=dict)
    regression: dict | None = None
    trace: dict | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_row(
    *,
    prompt_len: int,
    num_steps: int,
    e2e_samples: list[float],
    regression: dict | None,
    trace: dict | None,
    meta: dict,
    vision_probe_ms: float | None = None,
) -> PhaseRow:
    est = stats(e2e_samples)
    e2e_ms = est["median"]
    freq_hz = 1000.0 / e2e_ms if e2e_ms > 0 else float("nan")

    action_ms = vision_vlm_ms = vision_ms = vlm_ms = None
    notes_parts: list[str] = []

    # --- Action vs (Vision+VLM) from the regression (preferred) ---
    if regression is not None:
        action_ms = regression["slope_ms_per_step"] * num_steps
        vision_vlm_ms = regression["intercept_ms"]
        method = "regression"
        if regression.get("r2") is not None and regression["r2"] < 0.98:
            notes_parts.append(f"regression r2={regression['r2']:.3f} (low; results noisy)")
    elif trace is not None and trace.get("ok"):
        # fallback: use raw trace device-time buckets (note: device time, not wall)
        b = trace["buckets_ms_per_infer"]
        action_ms = b.get("action", 0.0)
        vision_vlm_ms = b.get("vision", 0.0) + b.get("vlm", 0.0)
        vision_ms = b.get("vision", 0.0)
        vlm_ms = b.get("vlm", 0.0)
        method = "trace-only"
        notes_parts.append("phase split from trace device-time (no regression)")
    else:
        method = "e2e-only"
        notes_parts.append("no regression and no usable trace: only E2E/Freq available")

    # --- Vision vs VLM split ---
    # Preferred: direct device-synced timing of the public SigLIP encoder (vision_probe_ms),
    # with VLM = (Vision+VLM total from regression) - Vision. The fused XLA trace cannot
    # separate img from llm ops, so this minimal probe is how we split the two.
    if method == "regression" and vision_probe_ms is not None and vision_vlm_ms is not None:
        vision_ms = vision_probe_ms
        vlm_ms = max(vision_vlm_ms - vision_probe_ms, 0.0)
        method = "regression+vision-probe"
        if vision_probe_ms > vision_vlm_ms:
            notes_parts.append(
                f"vision probe ({vision_probe_ms:.2f}ms) > regression intercept "
                f"({vision_vlm_ms:.2f}ms); VLM clamped to 0 (noisy/small model)"
            )
    elif method == "regression":
        notes_parts.append("Vision+VLM reported combined (no vision probe)")

    # --- percentages as share of (Vision+VLM+Action) ---
    action_pct = vision_pct = vlm_pct = None
    if action_ms is not None and vision_vlm_ms is not None:
        total = action_ms + vision_vlm_ms
        if total > 0:
            action_pct = 100.0 * action_ms / total
            if vision_ms is not None and vlm_ms is not None:
                vision_pct = 100.0 * vision_ms / total
                vlm_pct = 100.0 * vlm_ms / total
            # consistency check vs measured E2E
            disc = abs(total - e2e_ms) / e2e_ms * 100.0 if e2e_ms > 0 else float("nan")
            notes_parts.append(f"sum_of_phases={total:.2f}ms vs E2E={e2e_ms:.2f}ms ({disc:.1f}% diff)")

    return PhaseRow(
        prompt_len=prompt_len,
        num_steps=num_steps,
        e2e_ms=e2e_ms,
        freq_hz=freq_hz,
        action_ms=action_ms,
        vision_vlm_ms=vision_vlm_ms,
        vision_ms=vision_ms,
        vlm_ms=vlm_ms,
        action_pct=action_pct,
        vision_pct=vision_pct,
        vlm_pct=vlm_pct,
        method=method,
        notes="; ".join(notes_parts),
        e2e_stats=est,
        regression=regression,
        trace=trace,
        meta=meta,
    )
