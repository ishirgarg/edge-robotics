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
import-light `report` stage (mirrors roofline.py). All dims are reused from roofline.py.
"""

from __future__ import annotations

from .roofline import GEMMA_VARIANTS, SIGLIP, ModelShape, QuantScheme, _bytes_of


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
    pg = GEMMA_VARIANTS[shape.paligemma_variant]
    depth, kvh, hd, width = pg["depth"], pg["kv_heads"], pg["head_dim"], pg["width"]
    kv_b = _bytes_of(scheme.kv)

    kv_bytes = depth * 2 * shape.prefix_len * kvh * hd * shape.batch * kv_b   # K and V, all layers
    mask_bytes = shape.prefix_len * shape.batch                              # 1 byte/token pad mask
    state_bytes = shape.action_dim * shape.batch * 4                         # fp32 proprio state (pi0)
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
