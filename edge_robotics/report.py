"""Render the per-run summary and export JSON/CSV. One run == one (system, config, prompt_len)."""

from __future__ import annotations

import csv
import json
import os
from typing import Any


def _ms(d: dict | None, key: str = "median") -> str:
    if not d:
        return "—"
    v = d.get(key)
    return f"{v:.2f}" if v is not None else "—"


def _phase_cells(phases_ms: dict | None, phases_pct: dict | None, order: tuple[str, ...]) -> list[str]:
    out = []
    for p in order:
        if not phases_ms or phases_ms.get(p) is None:
            out.append("—")
            continue
        ms = phases_ms[p]
        pct = (phases_pct or {}).get(p)
        out.append(f"{ms:.2f} ({pct:.1f}%)" if pct is not None else f"{ms:.2f}")
    return out


def render_summary(meta: dict, result: dict) -> str:
    order = ("vision", "vlm", "action")
    lines: list[str] = []
    lines.append(
        f"Device: {meta.get('device_kind','?')} | system={meta.get('system','?')} | "
        f"config={meta.get('config_name','?')} | prompt_len={meta.get('prompt_len','?')} | "
        f"num_steps={meta.get('num_steps','?')} | batch={meta.get('batch_size','?')} | "
        f"iters={meta.get('iters','?')} | compile={meta.get('compile_mode','?')}"
    )
    lines.append("")

    seg = result.get("e2e_segmented_ms")
    nat = result.get("e2e_native_ms")
    seg_mode = meta.get("compile_mode")
    nat_mode = meta.get("native_compile_mode")
    seg_tag = f", {seg_mode}" if seg_mode else ""
    lines.append(f"E2E (segmented, headline{seg_tag}) : {_ms(seg)} ms   (p90 {_ms(seg,'p90')}, min {_ms(seg,'min')})")
    if nat:
        ov = result.get("segmentation_overhead_pct")
        # Only call the gap "segmentation overhead" when both paths used the SAME compile mode;
        # otherwise it also includes the mode difference, so label it neutrally.
        same_mode = (seg_mode == nat_mode) or (seg_mode is None and nat_mode is None)
        nat_tag = f", {nat_mode}" if nat_mode else ""
        if ov is not None:
            delta = f"  [segmentation {ov:+.1f}%]" if same_mode else f"  [Δ vs segmented {ov:+.1f}%, modes differ]"
        else:
            delta = ""
        lines.append(f"E2E (native repo-default fast path{nat_tag}): {_ms(nat)} ms{delta}")
    if result.get("freq_hz") is not None:
        lines.append(f"Freq                      : {result['freq_hz']:.2f} Hz")
    lines.append("")

    bd = result.get("breakdown_nvtx")
    if bd:
        cells = _phase_cells(bd["phases_ms_per_infer"], bd.get("phases_pct"), order)
        lines.append("Component breakdown (nsys NVTX GPU projection, graphs ON):")
        lines.append(f"  vision={cells[0]}  vlm={cells[1]}  action={cells[2]}")
        # Honest cross-checks: the per-phase GPU projection should sum to ~the E2E wall and ~the
        # total GPU kernel time. (Σphases can slightly exceed kernel-sum: projection counts the full
        # per-range GPU-busy span incl. memops/overheads, kernel-sum counts kernels only.)
        phases_sum = sum(v for v in bd["phases_ms_per_infer"].values() if v)
        gpu = bd.get("total_gpu_ms_per_infer")
        xs = [f"Σphases={phases_sum:.2f}ms"]
        if gpu:
            xs.append(f"nsys GPU kernels={gpu:.2f}ms")
        if seg and seg.get("median"):
            xs.append(f"E2E wall={seg['median']:.2f}ms")
        lines.append("  cross-check: " + "  |  ".join(xs))
    elif result.get("breakdown_nvtx_error"):
        lines.append(f"Component breakdown (NVTX): unavailable ({result['breakdown_nvtx_error']})")

    comp = result.get("components_standalone")
    if comp:
        cells = _phase_cells(comp["phases_ms_per_infer"], comp.get("phases_pct"), order)
        lines.append("Per-component standalone (graphs ON, isolated):")
        lines.append(f"  vision={cells[0]}  vlm={cells[1]}  action={cells[2]}")

    kb = result.get("kernel_buckets")
    if kb:
        b = kb["buckets_ms_per_infer"]
        pct = kb.get("buckets_pct", {})
        order_k = ("attention", "gemm", "conv", "elementwise", "other")
        parts = [f"{k}={b[k]:.2f}({pct.get(k,0):.0f}%)" for k in order_k if k in b]
        lines.append("Kernel-family buckets (nsys, graphs ON):")
        lines.append("  " + "  ".join(parts))
    elif result.get("kernel_buckets_error"):
        lines.append(f"Kernel buckets: unavailable ({result['kernel_buckets_error']})")

    return "\n".join(lines)


def write_outputs(output_prefix: str, meta: dict, result: dict) -> tuple[str, str]:
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)) or ".", exist_ok=True)
    json_path = f"{output_prefix}.json"
    csv_path = f"{output_prefix}.csv"

    with open(json_path, "w") as f:
        json.dump({"meta": meta, "result": result}, f, indent=2, default=_json_default)

    # Long-format CSV: one row per (metric, phase) for easy plotting across a sweep.
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "config_name", "prompt_len", "num_steps", "metric", "phase", "ms", "pct"])
        base = [meta.get("system"), meta.get("config_name"), meta.get("prompt_len"), meta.get("num_steps")]

        def emit(metric: str, phases_ms: dict | None, phases_pct: dict | None):
            if not phases_ms:
                return
            for k, v in phases_ms.items():
                pct = (phases_pct or {}).get(k)
                w.writerow(base + [metric, k, "" if v is None else f"{v:.4f}",
                                   "" if pct is None else f"{pct:.2f}"])

        seg = result.get("e2e_segmented_ms")
        if seg:
            w.writerow(base + ["e2e", "segmented", f"{seg['median']:.4f}", ""])
        nat = result.get("e2e_native_ms")
        if nat:
            w.writerow(base + ["e2e", "native", f"{nat['median']:.4f}", ""])
        if result.get("breakdown_nvtx"):
            emit("breakdown_nvtx", result["breakdown_nvtx"]["phases_ms_per_infer"],
                 result["breakdown_nvtx"].get("phases_pct"))
        if result.get("components_standalone"):
            emit("components_standalone", result["components_standalone"]["phases_ms_per_infer"],
                 result["components_standalone"].get("phases_pct"))
        if result.get("kernel_buckets"):
            emit("kernel_buckets", result["kernel_buckets"]["buckets_ms_per_infer"],
                 result["kernel_buckets"].get("buckets_pct"))

    return json_path, csv_path


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    return str(o)
