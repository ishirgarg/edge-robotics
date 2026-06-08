"""CLI config + orchestration for the edge-robotics profiler.

One invocation profiles ONE (system, config, prompt_len). Sweeps are the shell's job
(profile_sweep.sh). The model always runs in its real graphs-on form; attribution comes from
Nsight Systems. Because nsys must wrap the whole process and CUPTI perturbs wall timing, the work
is split across four cheap modes the shell drives in sequence:

  time   : load model, time segmented E2E (headline) + native E2E (cross-check) + per-component
           standalone -> <out>.timing.json. NOT under nsys, so the wall numbers are pristine.
  nsys   : load model, warmup, bracket ONE steady-state segmented inference with cudaProfilerStart/
           Stop + NVTX. Run by the shell UNDER `nsys profile`; nsys writes <out>.nsys-rep.
  parse  : no model load. Parse <out>.nsys-rep -> NVTX per-phase split + kernel-family buckets ->
           <out>.breakdown.json.
  report : no model load. Merge timing + breakdown -> <out>.json / <out>.csv and print a summary.

This module is import-light: defining Config must not import torch/openpi, so the entrypoint can
pin CUDA_VISIBLE_DEVICES before any CUDA context is created. Heavy imports live inside run().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# NVTX phase names each backend's segmented path emits (used by `parse` without loading the model).
# Empty tuple => opaque fused graph; breakdown degrades to kernel-family buckets only.
SYSTEM_PHASES: dict[str, tuple[str, ...]] = {
    "pi05_openpi_torch": ("vision", "vlm", "action"),
    "pi05_realtimevla": ("vision", "vlm", "action"),
}


@dataclass
class Config:
    # --- required identity flags ---
    system: Literal["pi05_realtimevla", "pi05_openpi_torch"]
    """Which system: 'pi05_openpi_torch' (openpi's native PyTorch port; full NVTX phase split) or
    'pi05_realtimevla' (dexmal/realtime-vla Triton; opaque graph, kernel-bucket split only)."""
    config_name: str
    """pi-0.5 config name (e.g. pi05_libero, pi05_droid, pi05_base, debug_pi05)."""
    checkpoint: str
    """Checkpoint: local path/gs:// URI, or the literal 'random' for random-init (no download)."""
    gpu: int
    """CUDA device index to pin."""

    # --- operational knobs ---
    mode: Literal["time", "time-native", "nsys", "parse", "report"] = "time"
    """Pipeline stage (see module docstring). The shell drives them in sequence. `time-native` is a
    SEPARATE process from `time` on purpose: openpi's single fused graph and the per-phase segmented
    graphs thrash the shared cudagraph pool if exercised together, so the native cross-check is run
    in isolation."""
    prompt_len: int = 200
    """max_token_len for this run (single value; sweeps are the shell's job)."""
    num_steps: int = 10
    """Flow-matching denoise steps."""
    warmup: int = 3
    """Warmup iterations (absorb torch.compile + cudagraph capture) before timing."""
    iters: int = 20
    """Timed iterations (also the #inferences bracketed for nsys)."""
    batch_size: int = 1
    """Inference batch size."""
    output: str = "out/profile"
    """Output prefix. Writes <prefix>.{timing,breakdown}.json, <prefix>.nsys-rep, <prefix>.{json,csv}."""
    nsys_rep: str | None = None
    """Path to the .nsys-rep for `parse` (default <output>.nsys-rep)."""


def _make_system(name: str):
    if name == "pi05_realtimevla":
        from .systems.pi05_realtimevla import Pi05RealtimeVlaSystem

        return Pi05RealtimeVlaSystem()
    if name == "pi05_openpi_torch":
        from .systems.pi05_openpi_torch import Pi05OpenpiTorchSystem

        return Pi05OpenpiTorchSystem()
    raise ValueError(f"unknown system '{name}'")


def _load(config: Config):
    system = _make_system(config.system)
    return system.load(
        config_name=config.config_name,
        checkpoint=config.checkpoint,
        prompt_len=config.prompt_len,
        num_steps=config.num_steps,
        batch_size=config.batch_size,
    )


def _environment_meta(config: Config) -> dict:
    """Full reproducibility manifest: hardware target, driver/toolkit, package versions, git state,
    the exact invocation, and the env vars that change results. Gathered once, in the `time` stage.
    Best-effort throughout — a missing tool degrades a field to None, never breaks a run."""
    import importlib.metadata as ilm
    import os
    import platform
    import socket
    import subprocess
    import sys
    import time

    import torch

    def _pkg(name: str):
        try:
            return ilm.version(name)
        except Exception:  # noqa: BLE001
            return None

    def _sh(cmd: list[str]):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_commit = _sh(["git", "-C", repo, "rev-parse", "HEAD"])
    git_dirty = bool(_sh(["git", "-C", repo, "status", "--porcelain"]))

    # GPU/driver: torch for the device torch actually uses; nvidia-smi for the host driver string.
    dev: dict = {}
    driver_api = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        dev = {
            "name": p.name,
            "compute_capability": f"{p.major}.{p.minor}",
            "total_memory_mib": round(p.total_memory / (1024**2)),
            "multi_processor_count": p.multi_processor_count,
            "uuid": str(getattr(p, "uuid", None)) if getattr(p, "uuid", None) else None,
        }
        try:
            driver_api = torch._C._cuda_getDriverVersion()  # noqa: SLF001  (e.g. 12060)
        except Exception:  # noqa: BLE001
            driver_api = None
    nvidia_driver = _sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if nvidia_driver:
        nvidia_driver = nvidia_driver.splitlines()[0].strip()

    from .profiling.nsys import nsys_bin

    nsys_path = nsys_bin()
    nsys_ver = None
    if nsys_path:
        v = _sh([nsys_path, "--version"])
        nsys_ver = v.splitlines()[0].strip() if v else None

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "invocation": " ".join(sys.argv),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git": {"commit": git_commit, "dirty": git_dirty, "repo": repo},
        "hardware": {
            "gpu_index_pinned": config.gpu,
            "device": dev,
            "nvidia_driver_version": nvidia_driver,
            "cuda_driver_api_version": driver_api,
            "cuda_toolkit_torch_built": torch.version.cuda,
        },
        "packages": {
            "torch": torch.__version__,
            "triton": _pkg("triton"),
            "transformers": _pkg("transformers"),
            "numpy": _pkg("numpy"),
            "openpi": _pkg("openpi"),
            "jax": _pkg("jax"),
            "jaxlib": _pkg("jaxlib"),
            "nsys": nsys_ver,
        },
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "env": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES", "OPENPI_TORCH_COMPILE_MODE", "OPENPI_TORCH_NATIVE_COMPILE_MODE",
                "CROSS_CHECK_NATIVE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_FX_GRAPH_CACHE", "JAX_PLATFORMS",
            )
        },
    }


def _top_meta(config: Config, loaded) -> dict:
    m = loaded.meta
    return {
        "system": config.system,
        "config_name": config.config_name,
        "checkpoint": m.get("checkpoint"),
        "device_kind": m.get("device_kind"),
        "n_devices": m.get("n_devices"),
        "prompt_len": config.prompt_len,
        "num_steps": config.num_steps,
        "batch_size": config.batch_size,
        "warmup": config.warmup,
        "iters": config.iters,
        "compile_mode": m.get("compile_mode"),
        "native_compile_mode": m.get("native_compile_mode"),
        "graphs_on": m.get("graphs_on"),
        "backend": m.get("backend"),
        "backend_version": m.get("torch_version"),
        "attribution": m.get("attribution", "n/a"),
        "nvtx_phases": list(SYSTEM_PHASES.get(config.system, ())),
        "model_weight_dtype": m.get("dtype"),  # e.g. "bfloat16" — surfaced for quick reproducibility
        "model": {
            k: m.get(k)
            for k in (
                "action_horizon", "action_dim", "paligemma_variant", "action_expert_variant",
                "dtype", "n_images", "prefix_len_nominal", "max_token_len",
            )
        },
        "environment": _environment_meta(config),
    }


def run(config: Config) -> None:
    dispatch = {
        "time": _run_time, "time-native": _run_time_native,
        "nsys": _run_nsys, "parse": _run_parse, "report": _run_report,
    }
    if config.mode not in dispatch:
        raise ValueError(f"unknown mode '{config.mode}'")
    dispatch[config.mode](config)


def _run_time(config: Config) -> None:
    """Pristine wall timing (no nsys): segmented E2E (headline) + per-component standalone.

    Deliberately does NOT exercise the native path — that shares the cudagraph pool with the
    per-phase graphs and would thrash. The native cross-check is its own process (`time-native`).
    """
    from .profiling.walltime import time_callable

    loaded = _load(config)
    print(f"[time] {config.system}/{config.config_name} L={config.prompt_len} — timing segmented E2E...")
    seg = time_callable(loaded.infer_segmented, loaded.block, warmup=config.warmup, iters=config.iters)

    comp = None
    if loaded.component_profiler is not None:
        print("[time] per-component standalone (graphs ON)...")
        comp = loaded.component_profiler(warmup=config.warmup, iters=config.iters)

    payload = {
        "meta": _top_meta(config, loaded),
        "e2e_segmented_ms": seg,
        "components_standalone": comp,
    }
    path = f"{config.output}.timing.json"
    _write_json(path, payload)
    sm = sorted(seg)[len(seg) // 2]
    print(f"[time] segmented median ~{sm:.2f} ms -> wrote {path}")


def _run_time_native(config: Config) -> None:
    """Cross-check ONLY: time openpi's native single-compile E2E, in isolation (own process).

    Run separately from `time` so the native fused graph never coexists with the per-phase graphs
    (which makes dynamo recompile every call). The native graph also needs more cudagraph warmup.
    """
    from .profiling.walltime import time_callable

    loaded = _load(config)
    if loaded.infer_native is None:
        print(f"[time-native] {config.system} has no native single-compile path; skipping.")
        _write_json(f"{config.output}.timing-native.json", {"e2e_native_ms": None})
        return
    nat_warmup = max(config.warmup, 8)
    print(f"[time-native] timing native single-compile E2E (warmup={nat_warmup})...")
    nat = time_callable(loaded.infer_native, loaded.block, warmup=nat_warmup, iters=config.iters)
    _write_json(f"{config.output}.timing-native.json", {"e2e_native_ms": nat})
    nm = sorted(nat)[len(nat) // 2]
    print(f"[time-native] native median ~{nm:.2f} ms")


def _run_nsys(config: Config) -> None:
    """Bracket exactly one steady-state run of the segmented E2E for nsys to capture."""
    from .profiling.nsys import profiler_capture, under_nsys

    if not under_nsys():
        print("[nsys] WARNING: not running under `nsys profile` — no capture will be produced. "
              "Launch via profile_one.sh (it wraps this process in nsys).")
    loaded = _load(config)
    print(f"[nsys] warmup x{config.warmup}, then capturing {config.iters} segmented inferences...")
    for _ in range(config.warmup):
        loaded.block(loaded.infer_segmented())
    with profiler_capture():
        for _ in range(config.iters):
            loaded.infer_segmented()
    print("[nsys] capture window closed.")


def _run_parse(config: Config) -> None:
    """Parse the nsys report into NVTX per-phase split + kernel-family buckets (no model load)."""
    from .profiling.nsys import parse_kernel_buckets, parse_nvtx_gpu_proj

    rep = config.nsys_rep or f"{config.output}.nsys-rep"
    phases = SYSTEM_PHASES.get(config.system, ())
    breakdown = parse_nvtx_gpu_proj(rep, iters=config.iters, phases=phases) if phases else None
    buckets = parse_kernel_buckets(rep, iters=config.iters)

    payload = {"breakdown_nvtx": breakdown, "kernel_buckets": buckets}
    path = f"{config.output}.breakdown.json"
    _write_json(path, payload)
    if breakdown and breakdown.get("ok"):
        p = breakdown["phases_ms_per_infer"]
        ssum = sum(v for v in p.values() if v)
        print(f"[parse] NVTX vision={p.get('vision',0):.2f} vlm={p.get('vlm',0):.2f} "
              f"action={p.get('action',0):.2f} ms (Σ={ssum:.2f} vs {breakdown.get('total_gpu_ms_per_infer',0):.2f}ms GPU kernels)")
    elif phases:
        print(f"[parse] NVTX breakdown unavailable: {(breakdown or {}).get('error')}")
    if buckets.get("ok"):
        print(f"[parse] kernel buckets: { {k: round(v,2) for k,v in buckets['buckets_ms_per_infer'].items()} }")
    print(f"[parse] wrote {path}")


def _run_report(config: Config) -> None:
    """Merge timing + breakdown into the final <output>.{json,csv} and print a summary."""
    import os

    from .metrics import build_result
    from .report import render_summary, write_outputs

    timing = _read_json(f"{config.output}.timing.json") or {}
    native = _read_json(f"{config.output}.timing-native.json") or {}
    breakdown = _read_json(f"{config.output}.breakdown.json") or {}
    meta = timing.get("meta") or {"system": config.system, "config_name": config.config_name,
                                  "prompt_len": config.prompt_len, "num_steps": config.num_steps,
                                  "batch_size": config.batch_size, "iters": config.iters}

    result = build_result(
        e2e_segmented=timing.get("e2e_segmented_ms"),
        e2e_native=native.get("e2e_native_ms"),
        breakdown_nvtx=breakdown.get("breakdown_nvtx"),
        kernel_buckets=breakdown.get("kernel_buckets"),
        components_standalone=timing.get("components_standalone"),
    )
    print("\n" + render_summary(meta, result))
    json_path, csv_path = write_outputs(config.output, meta, result)

    # Standalone reproducibility manifest (hardware target, driver/toolkit, package versions, git
    # state, dtype, model config, env vars, invocation) — same `meta` the results carry, written as
    # its own file so it travels with the run. Restores the old pi05_profile.sh config.json.
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(config.output)) or ".", "config.json")
    _write_json(cfg_path, meta)
    print(f"\n[report] wrote {json_path}, {csv_path}, and {cfg_path}")


def _write_json(path: str, payload: dict) -> None:
    import json
    import os

    from .report import _json_default

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _read_json(path: str) -> dict | None:
    import json
    import os

    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    """Console-script entry. Prefer scripts/profile_policy.py for GPU pinning."""
    import os

    import tyro

    config = tyro.cli(Config)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.gpu))
    run(config)


if __name__ == "__main__":
    main()
