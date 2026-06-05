"""Render the 5-metric table and export JSON/CSV."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from .metrics import PhaseRow


def _cell(ms: float | None, pct: float | None) -> str:
    if ms is None:
        return "—"
    if pct is None:
        return f"{ms:.2f}"
    return f"{ms:.2f} ({pct:.1f}%)"


def render_table(rows: list[PhaseRow], meta: dict) -> str:
    head = (
        f"Device: {meta.get('device_kind','?')} | system={meta.get('system','?')} | "
        f"config={meta.get('config_name','?')} | num_steps={meta.get('num_steps','?')} | "
        f"batch={meta.get('batch_size','?')} | iters={meta.get('iters','?')}"
    )
    cols = ["prompt_len", "Vision ms (%)", "VLM ms (%)", "Action ms (%)", "E2E ms", "Freq Hz"]
    widths = [10, 18, 18, 18, 10, 9]
    lines = [head, ""]
    lines.append("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        vlm_cell = _cell(r.vlm_ms, r.vlm_pct)
        if r.vlm_ms is None and r.vision_vlm_ms is not None:
            vlm_cell = f"{r.vision_vlm_ms:.2f} (V+VLM)"
        cells = [
            str(r.prompt_len),
            _cell(r.vision_ms, r.vision_pct),
            vlm_cell,
            _cell(r.action_ms, r.action_pct),
            f"{r.e2e_ms:.2f}",
            f"{r.freq_hz:.2f}",
        ]
        lines.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    # notes (per row, since method/discrepancy can vary)
    lines.append("")
    for r in rows:
        lines.append(f"  [L={r.prompt_len}] method={r.method}; {r.notes}")
    return "\n".join(lines)


def write_outputs(output_prefix: str, rows: list[PhaseRow], meta: dict) -> tuple[str, str]:
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)) or ".", exist_ok=True)
    json_path = f"{output_prefix}.json"
    csv_path = f"{output_prefix}.csv"

    with open(json_path, "w") as f:
        json.dump({"meta": meta, "rows": [r.to_dict() for r in rows]}, f, indent=2, default=_json_default)

    # CSV: one row per (prompt_len, phase) for easy plotting.
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_len", "num_steps", "phase", "ms", "pct", "e2e_ms", "freq_hz", "method"])
        for r in rows:
            phases = [
                ("vision", r.vision_ms, r.vision_pct),
                ("vlm", r.vlm_ms, r.vlm_pct),
                ("action", r.action_ms, r.action_pct),
            ]
            for name, ms, pct in phases:
                w.writerow(
                    [
                        r.prompt_len,
                        r.num_steps,
                        name,
                        "" if ms is None else f"{ms:.4f}",
                        "" if pct is None else f"{pct:.2f}",
                        f"{r.e2e_ms:.4f}",
                        f"{r.freq_hz:.4f}",
                        r.method,
                    ]
                )
            # also a combined Vision+VLM row when not split
            if r.vision_ms is None and r.vision_vlm_ms is not None:
                w.writerow(
                    [r.prompt_len, r.num_steps, "vision+vlm", f"{r.vision_vlm_ms:.4f}", "", f"{r.e2e_ms:.4f}", f"{r.freq_hz:.4f}", r.method]
                )
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
