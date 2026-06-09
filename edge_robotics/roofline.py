"""Analytic roofline model for pi-0.5 — the "ideal" lower bound we measure ourselves against.

Method (after NVlabs/vla-perf, "How Fast Can I Run My VLA?", arXiv 2602.18397):
each operator's latency lower bound is

    T_op = max( FLOPs_op / peak_compute ,  Bytes_op / peak_membw )

i.e. the op is either compute-bound (first term) or memory-bound (second). A component's lower
bound is the SUM of its operators' lower bounds (T_m = Σ T_op); E2E is the sum of components.
Arithmetic intensity OI = FLOPs/Bytes; the hardware ridge point is peak_compute/peak_membw — an op
is compute-bound iff OI > ridge. This is a HARDWARE-IDEAL bound: perfect kernels, perfect overlap,
no launch/sync overhead. The gap between this and the measured per-phase GPU time is "kernel
efficiency" (MFU/MBU); the gap between measured GPU time and wall time is system overhead.

Everything here is pure arithmetic from model dims + a hardware spec — NO torch/openpi import, so it
runs in the import-light `report` stage. Default precision is bf16 (2 bytes) with fp32 accumulate, so
the compute ceiling is the device's DENSE bf16 tensor rate (= sparse/2); a QuantScheme (read from the
run's meta) overrides the weight/activation/KV bytes and the tensor-core peak for quantized variants.

We model each transformer layer as the operators that actually dominate: the q/k/v/o linear
projections, the (GeGLU) MLP's three matmuls, and the attention score+context matmuls. SigLIP adds a
patch-embed conv and a multimodal projector. Norms/residuals are tiny and folded into an
"elementwise" byte term. We do NOT assume kernel fusion: each op reads its inputs+weights from HBM
and writes its output back (the standard, slightly pessimistic memory roofline).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --- model dims (mirror openpi/src/openpi/models/gemma.py get_config + the SigLIP So400m the HF
#     PaliGemma config builds). Hardcoded (not imported) to keep this module torch/openpi-free. ---
GEMMA_VARIANTS: dict[str, dict] = {
    # width, depth, mlp_dim, num_heads, num_kv_heads, head_dim
    "gemma_2b": dict(width=2048, depth=18, mlp=16384, heads=8, kv_heads=1, head_dim=256),
    "gemma_300m": dict(width=1024, depth=18, mlp=4096, heads=8, kv_heads=1, head_dim=256),
    "dummy": dict(width=64, depth=4, mlp=128, heads=8, kv_heads=1, head_dim=16),
}
# SigLIP So400m vision tower (PaliGemma default; intermediate_size overridden to 4304 in openpi).
SIGLIP = dict(width=1152, depth=27, mlp=4304, heads=16, head_dim=72, patch=14, image_px=224,
              channels=3, tokens=256, proj_out=2048)


@dataclass
class HardwareSpec:
    """The peaks that define the roofline ceilings. bf16_tflops is the DENSE bf16 tensor rate
    (sparse/2), the ceiling for a bf16 model with fp32 accumulate. The int8/fp8/int4 fields are
    optional dense tensor-core peaks for quantized compute; when None they fall back to the common
    multipliers (int8/fp8 = 2x bf16, int4 = 4x bf16)."""
    name: str
    bf16_tflops: float       # dense bf16 (fp32-accumulate) tensor-core peak
    mem_bw_gbps: float       # HBM/GDDR bandwidth
    int8_tops: float | None = None
    fp8_tflops: float | None = None
    int4_tops: float | None = None

    def peak_for(self, dtype: str) -> float:
        """Dense tensor-core peak (FLOP/s or OP/s) for a compute dtype. bf16/fp16/fp32 use the bf16
        tensor path; quantized peaks use explicit datasheet values if given, else the standard
        2x (int8/fp8) / 4x (int4) of bf16."""
        d = str(dtype).lower()
        if d in ("int8", "i8"):
            return (self.int8_tops or 2.0 * self.bf16_tflops) * 1e12
        if d in ("fp8", "float8", "e4m3", "e5m2"):
            return (self.fp8_tflops or 2.0 * self.bf16_tflops) * 1e12
        if d in ("int4", "i4", "nf4"):
            return (self.int4_tops or 4.0 * self.bf16_tflops) * 1e12
        return self.bf16_tflops * 1e12

    def ridge_for(self, compute_dtype: str) -> float:
        """Ridge-point arithmetic intensity (FLOP/byte) for a compute dtype: OI above => compute-bound."""
        return self.peak_for(compute_dtype) / (self.mem_bw_gbps * 1e9)

    @property
    def ridge_oi(self) -> float:
        """bf16 ridge-point arithmetic intensity (FLOP/byte)."""
        return self.ridge_for("bf16")


# Dense bf16 tensor TFLOPS (= datasheet sparse / 2 for the GeForce/workstation fp32-accum parts;
# A100 has no fp32-accum penalty) and memory bandwidth. A6000 is this study's machine; 4090 (edge/
# consumer) and A100 (datacenter) are kept so the same roofline can be evaluated for other targets
# — central to "how far from ideal at the edge". Any other GPU: override via env (below).
HARDWARE: dict[str, HardwareSpec] = {
    "RTX A6000": HardwareSpec("RTX A6000", bf16_tflops=154.8, mem_bw_gbps=768.0),
    "RTX 4090": HardwareSpec("RTX 4090", bf16_tflops=165.2, mem_bw_gbps=1008.0),
    "A100": HardwareSpec("A100-SXM 80GB", bf16_tflops=312.0, mem_bw_gbps=2039.0),
}


def hardware_for(device_name: str | None) -> HardwareSpec:
    """Resolve a HardwareSpec from a torch device name; env overrides win (other-GPU one-offs).

    An UNKNOWN device with no env override warns loudly before defaulting to the A6000 peaks — scoring
    MFU/MBU/ideal-ms against the wrong ceiling would silently misstate "how far from ideal"."""
    base = None
    if device_name:
        for key, spec in HARDWARE.items():
            if key.lower() in device_name.lower():
                base = spec
                break
    bf16 = os.environ.get("EDGE_ROBOTICS_PEAK_BF16_TFLOPS")
    bw = os.environ.get("EDGE_ROBOTICS_MEM_BW_GBPS")
    if base is None and not (bf16 or bw):
        import sys
        print(f"[roofline] WARNING: unknown GPU {device_name!r}; defaulting to RTX A6000 peaks "
              "(154.8 TFLOPS / 768 GB/s). Set EDGE_ROBOTICS_PEAK_BF16_TFLOPS / EDGE_ROBOTICS_MEM_BW_GBPS "
              "to score against the real device.", file=sys.stderr)
    if base is None:
        base = HARDWARE["RTX A6000"]  # documented default (this study's machine)
    if bf16 or bw:
        base = HardwareSpec(base.name + " (overridden)", float(bf16) if bf16 else base.bf16_tflops,
                            float(bw) if bw else base.mem_bw_gbps,
                            int8_tops=base.int8_tops, fp8_tflops=base.fp8_tflops, int4_tops=base.int4_tops)
    return base


# ---------------------------------------------------------------------------------------------
# Dtype / quantization. The roofline defaults to bf16 everywhere (byte-identical to the model we
# ship today). A QuantScheme lets a future quantized run change weight / activation / KV bytes and
# the compute peak INDEPENDENTLY — e.g. W4A16 (int4 weights, bf16 activations: halves weight traffic,
# bf16 compute) or W8A8 (int8 weights+acts: halves both AND ~doubles the tensor-core peak). adaRMS
# denses stay fp32 regardless (they are not quantized in practice).
# ---------------------------------------------------------------------------------------------
_DTYPE_BYTES: dict[str, float] = {
    "fp32": 4, "float32": 4, "tf32": 4,
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2, "half": 2,
    "fp8": 1, "float8": 1, "e4m3": 1, "e5m2": 1, "int8": 1, "i8": 1,
    "int4": 0.5, "i4": 0.5, "nf4": 0.5,
}


def _bytes_of(dtype: str) -> float:
    """Bytes per element for a dtype name (defaults to bf16=2 for anything unrecognized)."""
    return _DTYPE_BYTES.get(str(dtype).lower(), 2.0)


@dataclass(frozen=True)
class QuantScheme:
    """Precision of the matmul-heavy path. Defaults = all bf16 (today's model, byte-identical).

    `compute` selects the tensor-core peak (a W8A8 run computes on int8 cores at ~2x bf16);
    `weights` sets GEMM weight-read bytes; `activations` sets activation/IO bytes; `kv` sets
    KV-cache bytes (often left bf16 even when weights are int4 — the asymmetry the roofline captures).
    NOTE: this scales the dominant q/k/v/o + MLP GEMMs and attention IO; adaRMS denses stay fp32 and
    the tiny patch-embed / norm terms are left at activation precision (negligible)."""
    weights: str = "bf16"
    activations: str = "bf16"
    kv: str = "bf16"
    compute: str = "bf16"

    @classmethod
    def from_meta(cls, meta: dict) -> "QuantScheme":
        # A run may carry a full {"weights","activations","kv","compute"} dict under meta["quant"];
        # otherwise fall back to the single compute_dtype (bf16 today) for every field.
        q = meta.get("quant") or {}
        cd = meta.get("compute_dtype") or (meta.get("model") or {}).get("compute_dtype") or "bf16"
        return cls(weights=q.get("weights", cd), activations=q.get("activations", cd),
                   kv=q.get("kv", cd), compute=q.get("compute", cd))


# ---------------------------------------------------------------------------------------------
# Operators. Each returns {flops, bytes, kind}. bytes = inputs + weights read + output written
# (no-fusion memory model). Dtypes default to bf16 (2 bytes); a QuantScheme overrides them.
# ---------------------------------------------------------------------------------------------
def _linear(tokens: int, in_dim: int, out_dim: int, *, batch: int = 1,
            w_dtype: str = "bf16", act_dtype: str = "bf16") -> dict:
    # bytes = input+output activations (act_dtype) + weights read once (w_dtype); no-fusion model.
    flops = 2.0 * batch * tokens * in_dim * out_dim
    act = batch * tokens * (in_dim + out_dim) * _bytes_of(act_dtype)
    return {"flops": flops, "bytes": act + in_dim * out_dim * _bytes_of(w_dtype), "kind": "gemm"}


def _attention(tokens_q: int, len_kv: int, heads: int, kv_heads: int, head_dim: int, *, batch: int = 1,
               act_dtype: str = "bf16", kv_dtype: str = "bf16") -> dict:
    # flash-style: QK^T then softmax·V; scores are not written to HBM.
    flops = 4.0 * batch * heads * tokens_q * len_kv * head_dim
    ab = _bytes_of(act_dtype)
    q = batch * tokens_q * heads * head_dim * ab
    kv = batch * len_kv * kv_heads * head_dim * 2 * _bytes_of(kv_dtype)  # K and V (MQA: kv_heads=1)
    o = batch * tokens_q * heads * head_dim * ab
    return {"flops": flops, "bytes": q + kv + o, "kind": "attention"}


def _elementwise(tokens: int, width: int, *, batch: int = 1, passes: int = 1, act_dtype: str = "bf16") -> dict:
    # norms/residuals/activations: ~0 FLOPs vs memory; read+write `passes` times.
    return {"flops": 0.0, "bytes": batch * tokens * width * _bytes_of(act_dtype) * 2 * passes,
            "kind": "elementwise"}


def _adarms_dense(width: int, cond_dim: int, *, batch: int = 1) -> dict:
    # pi05 action expert uses adaptive RMSNorm: a Linear(cond_dim -> 3*width, bias) kept in FP32
    # (4 bytes), applied to the 1-token time conditioning. Tiny FLOPs, but its WEIGHTS are re-read
    # from HBM every denoise step — material for the memory-bound action phase (see modeling_gemma).
    out = 3 * width
    return {"flops": 2.0 * batch * cond_dim * out,
            "bytes": cond_dim * out * 4 + batch * (cond_dim + out) * 4, "kind": "elementwise"}


def _decoder_layer_ops(tokens: int, len_kv: int, cfg: dict, *, batch: int = 1,
                       adarms_cond_dim: int | None = None,
                       w_dtype: str = "bf16", act_dtype: str = "bf16", kv_dtype: str = "bf16") -> list[dict]:
    """One gemma/paligemma decoder layer over `tokens` queries attending to `len_kv` keys.

    `adarms_cond_dim` (set for the pi05 action expert) adds the two per-layer adaRMS dense weights."""
    d, H, K, hd, m = cfg["width"], cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["mlp"]
    qhd, kvhd = H * hd, K * hd

    def lin(t, i, o):
        return _linear(t, i, o, batch=batch, w_dtype=w_dtype, act_dtype=act_dtype)

    ops = [
        {"name": "q_proj", **lin(tokens, d, qhd)},
        {"name": "k_proj", **lin(tokens, d, kvhd)},
        {"name": "v_proj", **lin(tokens, d, kvhd)},
        {"name": "attn", **_attention(tokens, len_kv, H, K, hd, batch=batch, act_dtype=act_dtype, kv_dtype=kv_dtype)},
        {"name": "o_proj", **lin(tokens, qhd, d)},
        {"name": "mlp_gate", **lin(tokens, d, m)},
        {"name": "mlp_up", **lin(tokens, d, m)},
        {"name": "mlp_down", **lin(tokens, m, d)},
        {"name": "norms", **_elementwise(tokens, d, batch=batch, passes=2, act_dtype=act_dtype)},
    ]
    if adarms_cond_dim is not None:  # 2 adaRMS modulation denses per layer (input + post-attn norm)
        ops += [{"name": "adarms_input", **_adarms_dense(d, adarms_cond_dim, batch=batch)},
                {"name": "adarms_postattn", **_adarms_dense(d, adarms_cond_dim, batch=batch)}]
    return ops


def _siglip_ops(batch: int = 1, *, w_dtype: str = "bf16", act_dtype: str = "bf16") -> list[dict]:
    s = SIGLIP
    T, d, H, hd, m = s["tokens"], s["width"], s["heads"], s["head_dim"], s["mlp"]

    def lin(t, i, o):
        return _linear(t, i, o, batch=batch, w_dtype=w_dtype, act_dtype=act_dtype)

    ops: list[dict] = [
        # patch-embed conv as a GEMM: each of T patches is (channels*patch*patch) -> width.
        {"name": "patch_embed", **lin(T, s["channels"] * s["patch"] ** 2, d)},
    ]
    for _ in range(s["depth"]):
        ops += [
            {"name": "q_proj", **lin(T, d, H * hd)},
            {"name": "k_proj", **lin(T, d, H * hd)},
            {"name": "v_proj", **lin(T, d, H * hd)},
            # full MHA, bidirectional; no KV cache, so KV bytes ride the activation dtype.
            {"name": "attn", **_attention(T, T, H, H, hd, batch=batch, act_dtype=act_dtype, kv_dtype=act_dtype)},
            {"name": "o_proj", **lin(T, H * hd, d)},
            {"name": "mlp_fc1", **lin(T, d, m)},
            {"name": "mlp_fc2", **lin(T, m, d)},
            {"name": "norms", **_elementwise(T, d, batch=batch, passes=2, act_dtype=act_dtype)},
        ]
    # multimodal projector: width -> paligemma width (2048), per token.
    ops.append({"name": "mm_projector", **lin(T, d, s["proj_out"])})
    return ops


@dataclass
class ModelShape:
    """Everything the roofline needs, read from a run's meta."""
    paligemma_variant: str
    action_expert_variant: str
    n_images: int
    prefix_len: int          # image tokens + padded language tokens (what the LLM actually processes)
    action_horizon: int
    num_steps: int
    pi05: bool = True         # pi05 action expert uses adaRMS (extra fp32 weights re-read per step)
    action_dim: int = 32      # state/action width; pi0 (non-pi05) projects state into a suffix token
    batch: int = 1

    @classmethod
    def from_meta(cls, meta: dict) -> "ModelShape":
        # `.get(...) or default` (not `.get(k, default)`): meta may store an explicit null for a key.
        m = meta.get("model") or {}
        n_images = int(m.get("n_images") or 3)
        prefix = m.get("prefix_len_nominal")
        if prefix is None:
            prefix = n_images * SIGLIP["tokens"] + int(meta.get("prompt_len") or 200)
        pi05 = m.get("pi05")
        if pi05 is None:  # fall back to the config-name convention (pi05_*, debug_pi05)
            pi05 = str(meta.get("config_name", "")).startswith(("pi05", "debug_pi05"))
        return cls(
            paligemma_variant=m.get("paligemma_variant") or "gemma_2b",
            action_expert_variant=m.get("action_expert_variant") or "gemma_300m",
            n_images=n_images,
            prefix_len=int(prefix),
            action_horizon=int(m.get("action_horizon") or 10),
            num_steps=int(meta.get("num_steps") or 10),
            pi05=bool(pi05),
            action_dim=int(m.get("action_dim") or 32),
            batch=int(meta.get("batch_size") or 1),
        )


def _phase_roofline(ops: list[dict], hw: HardwareSpec, *, repeat: int = 1, compute_dtype: str = "bf16") -> dict:
    """Aggregate operators: per-op T = max(compute, memory); phase T = repeat * Σ T_op.

    Keeps t_compute_ms / t_memory_ms (the two roofline terms) alongside t_roofline_ms — together they
    show WHY a phase lands compute- or memory-bound (the larger term wins per op). `compute_dtype`
    selects the tensor-core peak (bf16 by default; a quantized run uses the int8/fp8/int4 peak)."""
    peak_f = hw.peak_for(compute_dtype)
    peak_b = hw.mem_bw_gbps * 1e9
    tot_flops = tot_bytes = t_compute = t_mem = t_roof = 0.0
    for op in ops:
        f, b = op["flops"], op["bytes"]
        tc, tm = f / peak_f, b / peak_b
        tot_flops += f; tot_bytes += b; t_compute += tc; t_mem += tm; t_roof += max(tc, tm)
    oi = tot_flops / tot_bytes if tot_bytes else 0.0
    return {
        "flops": tot_flops * repeat,
        "bytes": tot_bytes * repeat,
        "arithmetic_intensity": oi,
        "bound": "compute" if oi > hw.ridge_for(compute_dtype) else "memory",
        "t_compute_ms": t_compute * 1e3 * repeat,
        "t_memory_ms": t_mem * 1e3 * repeat,
        "t_roofline_ms": t_roof * 1e3 * repeat,
    }


def compute_roofline(shape: ModelShape, hw: HardwareSpec, scheme: QuantScheme | None = None) -> dict:
    """Per-phase + E2E roofline lower bounds for one pi-0.5 inference."""
    scheme = scheme or QuantScheme()
    wd, ad, kvd, cd = scheme.weights, scheme.activations, scheme.kv, scheme.compute
    pg = GEMMA_VARIANTS[shape.paligemma_variant]
    ax = GEMMA_VARIANTS[shape.action_expert_variant]

    # Vision: SigLIP on every image (masked images are still encoded), batched as batch*n_images.
    vision_ops = _siglip_ops(batch=shape.batch * shape.n_images, w_dtype=wd, act_dtype=ad)
    vision = _phase_roofline(vision_ops, hw, compute_dtype=cd)

    # VLM: gemma_2b prefill over the full prefix (self-attention, len_kv = prefix_len). The KV cache
    # write IS already counted: k_proj/v_proj's output-write bytes ARE the K,V written to HBM.
    vlm_ops: list[dict] = []
    for _ in range(pg["depth"]):
        vlm_ops += _decoder_layer_ops(shape.prefix_len, shape.prefix_len, pg, batch=shape.batch,
                                      w_dtype=wd, act_dtype=ad, kv_dtype=kvd)
    vlm = _phase_roofline(vlm_ops, hw, compute_dtype=cd)

    # Action: gemma_300m, num_steps denoise iterations. The suffix is `action_horizon` query tokens,
    # PLUS one continuous state token for pi0 (embed_suffix's state_proj; pi05 has NO such token —
    # state is discretized into the prompt instead). Each step re-reads ALL expert weights; the prefix
    # KV-cache READ is already in _attention's `kv` term over the full len_kv (not added again).
    suffix_len = shape.action_horizon + (0 if shape.pi05 else 1)
    len_kv = shape.prefix_len + suffix_len
    adarms = ax["width"] if shape.pi05 else None  # action expert uses adaRMS when pi05
    step_ops: list[dict] = []
    for _ in range(ax["depth"]):
        step_ops += _decoder_layer_ops(suffix_len, len_kv, ax, batch=shape.batch,
                                       adarms_cond_dim=adarms, w_dtype=wd, act_dtype=ad, kv_dtype=kvd)
    if adarms is not None:  # final model.norm is also an adaRMS dense
        step_ops.append({"name": "adarms_final", **_adarms_dense(ax["width"], adarms, batch=shape.batch)})
    else:  # pi0: the continuous state token is projected into the suffix (state_proj) every step
        step_ops.append({"name": "state_proj", **_linear(1, shape.action_dim, ax["width"],
                                                         batch=shape.batch, w_dtype=wd, act_dtype=ad)})
    action = _phase_roofline(step_ops, hw, repeat=shape.num_steps, compute_dtype=cd)

    e2e_ms = vision["t_roofline_ms"] + vlm["t_roofline_ms"] + action["t_roofline_ms"]
    e2e_flops = vision["flops"] + vlm["flops"] + action["flops"]
    e2e_bytes = vision["bytes"] + vlm["bytes"] + action["bytes"]
    return {
        "hardware": {"name": hw.name, "bf16_tflops": hw.bf16_tflops, "mem_bw_gbps": hw.mem_bw_gbps,
                     "ridge_point_oi": hw.ridge_oi},
        "shape": vars(shape),
        "phases": {"vision": vision, "vlm": vlm, "action": action},
        "e2e": {
            "t_roofline_ms": e2e_ms, "freq_hz_ideal": 1000.0 / e2e_ms if e2e_ms else None,
            "flops": e2e_flops, "bytes": e2e_bytes,
            "arithmetic_intensity": e2e_flops / e2e_bytes if e2e_bytes else 0.0,
        },
    }


def merge_with_measured(roofline: dict, *, phases_gpu_ms: dict | None, e2e_wall_ms: float | None,
                        hw: HardwareSpec, compute_dtype: str = "bf16") -> dict:
    """Attach efficiency vs measured: MFU/MBU and roofline/achieved ratio per phase.

    `phases_gpu_ms` is the nsys NVTX per-phase GPU time (the right "achieved" — pure device time).
    MFU = achieved FLOP/s / peak; MBU = achieved byte/s / peak. The peak uses the run's compute dtype,
    so MFU stays on [0,1] under quantization (a W8A8 run is measured against the int8 peak, not bf16).
    roofline_ms/achieved_ms ("efficiency") is what fraction of ideal we hit; 1.0 == on the roofline. We
    also report the achieved E2E vs the ideal lower bound, and (if wall) the system-overhead gap.
    """
    peak_f = hw.peak_for(compute_dtype)
    peak_b = hw.mem_bw_gbps * 1e9
    out = {"per_phase": {}, "e2e": {}}
    gpu_sum = 0.0
    for ph, r in roofline["phases"].items():
        meas_ms = (phases_gpu_ms or {}).get(ph)
        entry = {"roofline_ms": r["t_roofline_ms"], "measured_gpu_ms": meas_ms, "bound": r["bound"],
                 "arithmetic_intensity": r["arithmetic_intensity"]}
        if meas_ms and meas_ms > 0:
            gpu_sum += meas_ms
            s = meas_ms / 1e3
            entry["mfu"] = (r["flops"] / s) / peak_f
            entry["mbu"] = (r["bytes"] / s) / peak_b
            # ideal/achieved — normally in (0,1]; can exceed 1 only if the (pessimistic, no-fusion)
            # byte estimate over-counts a heavily-fused memory-bound kernel, which flags that.
            entry["efficiency"] = r["t_roofline_ms"] / meas_ms
        out["per_phase"][ph] = entry
    ideal = roofline["e2e"]["t_roofline_ms"]
    e2e = {"roofline_ms": ideal, "measured_gpu_ms": gpu_sum or None,
           "freq_hz_ideal": roofline["e2e"]["freq_hz_ideal"]}
    if gpu_sum:
        e2e["gpu_efficiency"] = ideal / gpu_sum
    if e2e_wall_ms:
        e2e["measured_wall_ms"] = e2e_wall_ms
        e2e["roofline_vs_wall"] = ideal / e2e_wall_ms
        if gpu_sum:
            e2e["system_overhead_ms"] = e2e_wall_ms - gpu_sum
            e2e["system_overhead_pct"] = 100.0 * (e2e_wall_ms - gpu_sum) / e2e_wall_ms
    out["e2e"] = e2e
    return out


def analyze(meta: dict, *, phases_gpu_ms: dict | None = None, e2e_wall_ms: float | None = None) -> dict:
    """Top-level: build the roofline from a run's meta and merge measured efficiency if available."""
    shape = ModelShape.from_meta(meta)
    scheme = QuantScheme.from_meta(meta)
    env_dev = ((meta.get("environment") or {}).get("hardware") or {}).get("device") or {}
    hw = hardware_for(meta.get("device_kind") or env_dev.get("name"))
    rf = compute_roofline(shape, hw, scheme)
    rf["quant"] = vars(scheme)
    rf["measured"] = merge_with_measured(rf, phases_gpu_ms=phases_gpu_ms, e2e_wall_ms=e2e_wall_ms,
                                         hw=hw, compute_dtype=scheme.compute)
    return rf
