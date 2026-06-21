"""Server↔edge transfer sizing for a disaggregated VLA.

Motivating scenario: the heavy VLM (SigLIP vision tower + gemma_2b prefill) runs on a SERVER, and
only the lightweight action expert runs on the EDGE robot. Per inference, the data that must cross
the network is whatever the action expert CONDITIONS on — dominated by the prefix KV cache the
expert cross-attends to (gemma_2b produces it during prefill), plus the prefix pad mask and, for
pi0, the proprioceptive state token. We size that exactly from the model dims; the openpi-torch
backend also records a MEASURED KV-cache byte count from the real tensors as a cross-check.

This answers "how much network bandwidth is needed to run the VLM on server and the VLA on edge":
- `total_conditioning_bytes` = the per-inference transfer size (the headline number).
- `required_bandwidth_*` (if a freq is given) = that size × the inference rate.

Pure arithmetic from model dims + a QuantScheme — NO torch/openpi import, so it runs in the
import-light `report` stage.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- model dims (mirror openpi/src/openpi/models/gemma.py get_config + the SigLIP So400m the HF
#     PaliGemma config builds). Hardcoded (not imported) to keep this module torch/openpi-free. ---
GEMMA_VARIANTS: dict[str, dict] = {
    # width, depth, num_kv_heads, head_dim (the dims the KV-cache transfer sizing reads)
    "gemma_2b": dict(width=2048, depth=18, kv_heads=1, head_dim=256),
    "gemma_300m": dict(width=1024, depth=18, kv_heads=1, head_dim=256),
    "dummy": dict(width=64, depth=4, kv_heads=1, head_dim=16),
}
# SigLIP So400m vision tower (PaliGemma default): 256 patch tokens per image.
SIGLIP = dict(tokens=256)

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
    """Precision of the data crossing the wire. Defaults = all bf16 (today's model, byte-identical).

    `kv` sets the prefix KV-cache bytes (the dominant term); `activations` sets the image-embedding
    bytes of the alternative vision-on-server split."""
    activations: str = "bf16"
    kv: str = "bf16"

    @classmethod
    def from_meta(cls, meta: dict) -> "QuantScheme":
        # A run may carry a full {"weights","activations","kv","compute"} dict under meta["quant"];
        # otherwise fall back to the single compute_dtype (bf16 today) for every field.
        q = meta.get("quant") or {}
        cd = meta.get("compute_dtype") or (meta.get("model") or {}).get("compute_dtype") or "bf16"
        return cls(activations=q.get("activations", cd), kv=q.get("kv", cd))


@dataclass
class ModelShape:
    """The model dims the transfer sizing needs, read from a run's meta."""
    paligemma_variant: str
    n_images: int
    prefix_len: int          # image tokens + padded language tokens (the gemma_2b prefill length)
    pi05: bool = True         # pi0 ships a continuous state token; pi05 folds state into the prompt
    action_dim: int = 32      # state/action width; pi0 projects state into a suffix token
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
            n_images=n_images,
            prefix_len=int(prefix),
            pi05=bool(pi05),
            action_dim=int(m.get("action_dim") or 32),
            batch=int(meta.get("batch_size") or 1),
        )


def conditioning_transfer(shape: ModelShape, scheme: QuantScheme | None = None, *,
                          freq_hz: float | None = None,
                          kv_cache_bytes_measured: int | None = None) -> dict:
    """Bytes that cross server->edge per inference for the VLM-on-server / action-expert-on-edge split.

    The prefix KV cache is the gemma_2b prefill output the expert cross-attends to: `depth` layers x
    {K,V} x [prefix_len, kv_heads, head_dim] (MQA => kv_heads=1, so this is far smaller than a full
    MHA cache). `kv` dtype controls its bytes. Also reports the ALTERNATIVE split (vision on server,
    VLM+action on edge => ship SigLIP image embeddings instead) for comparison.
    """
    scheme = scheme or QuantScheme()
    pg = GEMMA_VARIANTS.get(shape.paligemma_variant, GEMMA_VARIANTS["gemma_2b"])
    depth, kvh, hd, width = pg["depth"], pg["kv_heads"], pg["head_dim"], pg["width"]
    kv_b = _bytes_of(scheme.kv)

    kv_bytes = depth * 2 * shape.prefix_len * kvh * hd * shape.batch * kv_b   # K and V, all layers
    mask_bytes = shape.prefix_len * shape.batch                              # 1 byte/token pad mask
    # Only pi0 ships a raw state token (its continuous suffix token). pi05 carries state IN the prompt
    # (already inside the KV cache) or not at all (pi05_libero), so nothing extra crosses the wire.
    state_bytes = (shape.action_dim * shape.batch * 4) if not shape.pi05 else 0
    total = kv_bytes + mask_bytes + state_bytes
    # Alternative seam: vision on server, gemma_2b prefill + action on edge -> ship image embeddings.
    vision_embeds_bytes = shape.n_images * SIGLIP["tokens"] * width * shape.batch * _bytes_of(scheme.activations)

    out: dict = {
        "scenario": "VLM(server) -> action-expert(edge): transfer = prefix KV cache + pad mask + state",
        "kv_cache_bytes": int(kv_bytes),
        "kv_cache_mib": kv_bytes / 2**20,
        "prefix_pad_mask_bytes": int(mask_bytes),
        "state_bytes": int(state_bytes),
        "total_conditioning_bytes": int(total),
        "total_conditioning_mib": total / 2**20,
        "kv_cache_detail": {"layers": depth, "kv_heads": kvh, "head_dim": hd,
                            "prefix_len": shape.prefix_len, "dtype": scheme.kv, "bytes_per_el": kv_b},
        "alt_vision_split_bytes": int(vision_embeds_bytes),
        "alt_vision_split_mib": vision_embeds_bytes / 2**20,
    }
    if kv_cache_bytes_measured:
        out["kv_cache_bytes_measured"] = int(kv_cache_bytes_measured)
        out["kv_cache_measured_mib"] = kv_cache_bytes_measured / 2**20
        out["kv_analytic_over_measured"] = kv_bytes / kv_cache_bytes_measured
    if freq_hz:
        out["freq_hz"] = freq_hz
        out["required_bandwidth_mbytes_per_s"] = total / 1e6 * freq_hz
        out["required_bandwidth_mbit_per_s"] = total * 8 / 1e6 * freq_hz
    return out


def analyze(meta: dict, *, freq_hz: float | None = None) -> dict:
    """Top-level: size the server->edge transfer from a run's meta (+ a measured KV count if present)."""
    shape = ModelShape.from_meta(meta)
    scheme = QuantScheme.from_meta(meta)
    measured = meta.get("kv_cache_bytes_measured") or (meta.get("model") or {}).get("kv_cache_bytes_measured")
    return conditioning_transfer(shape, scheme, freq_hz=freq_hz, kv_cache_bytes_measured=measured)
