"""Empirically measure DROID input-image sizes in bytes.

Two things we care about:
  1) Bytes actually transmitted over the network today. The openpi client serializes
     numpy arrays with msgpack via `obj.tobytes()` (msgpack_numpy.py) -- i.e. RAW,
     uncompressed. So on-wire size == H*W*3 bytes/image at the resolution sent.
     DROID deployment resizes camera frames to 224x224 before sending (droid main.py).
  2) Bytes if we instead applied standard image compression (JPEG / PNG / WebP) per frame.

We load REAL DROID frames from the public lerobot/droid_100 dataset (same loader the
attention study uses) and measure both at native decode resolution and at the 224x224
inference resolution that is what's actually sent to the policy server.
"""
import io
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge_robotics.libero_obs import _decode_video_frame, _hf_get

# DROID camera columns actually fed to the policy (base exterior + wrist, left stereo only).
CAMS = ["observation.images.exterior_image_1_left", "observation.images.wrist_image_left"]
N_FRAMES = 30          # images to average over (across both cameras)
EPISODES = [0, 1, 2]   # spread across a few episodes


def resize_224(img: np.ndarray) -> np.ndarray:
    """Match openpi resize_with_pad: keep aspect, bilinear, zero-pad to 224x224."""
    h, w = img.shape[:2]
    r = min(224 / h, 224 / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((224, 224, 3), dtype=np.uint8)
    t, l = (224 - nh) // 2, (224 - nw) // 2
    out[t:t + nh, l:l + nw] = resized
    return out


def encoded_bytes(img: np.ndarray, fmt: str, **kw) -> int:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt, **kw)
    return buf.tell()


def collect_frames():
    import pandas as pd
    repo = "lerobot/droid_100"
    root = "/scratch/ishirgarg/openpi_cache/real_frames/droid"
    pq = _hf_get(repo, "data/chunk-000/file-000.parquet", os.path.join(root, "data.parquet"))
    df = pd.read_parquet(pq).reset_index(drop=True)
    mp4s = {c: _hf_get(repo, f"videos/{c}/chunk-000/file-000.mp4",
                       os.path.join(root, c.split(".")[-1] + ".mp4")) for c in CAMS}

    frames = []
    for ep in EPISODES:
        ep_rows = df.index[df["episode_index"] == ep]
        for k in range(0, 40, 4):
            if len(frames) >= N_FRAMES:
                return frames
            row_pos = int(ep_rows[min(k, len(ep_rows) - 1)])  # position in file == video frame index
            for cam in CAMS:
                img = _decode_video_frame(mp4s[cam], row_pos)
                frames.append((f"ep{ep}_fr{k}_{cam.split('.')[-1]}", np.asarray(img, dtype=np.uint8)))
    return frames


def summarize(name, sizes_raw, encoders):
    n = len(sizes_raw)
    raw = np.mean(sizes_raw)
    print(f"\n=== {name}  (n={n} images) ===")
    print(f"  raw uint8 (== bytes sent on wire today): {raw:9.0f} B/img  ({raw/1024:6.1f} KiB)")
    for label, sizes in encoders.items():
        m = np.mean(sizes)
        print(f"  {label:<22}: {m:9.0f} B/img  ({m/1024:6.1f} KiB)   "
              f"{raw/m:5.1f}x smaller   ({100*m/raw:4.1f}% of raw)")


def _timeit(fn, n_warmup=5, n_iter=50):
    """Median per-call latency in ms (median is robust to scheduler jitter)."""
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


# Optimized codecs (libjpeg-turbo via OpenCV C++ path) vs the PIL baseline used for the size table.
def _enc(name):
    if name == "PIL JPEG q90":
        return lambda im: (lambda b: (Image.fromarray(im).save(b, "JPEG", quality=90), b.getvalue())[1])(io.BytesIO())
    if name == "cv2 JPEG q90":
        return lambda im: cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
    if name == "cv2 WebP q80":
        return lambda im: cv2.imencode(".webp", im, [cv2.IMWRITE_WEBP_QUALITY, 80])[1]


def _dec(name):
    if name == "PIL JPEG q90":
        return lambda buf: np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))
    return lambda buf: cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def benchmark_latency(frames):
    """Per-image encode+decode latency at 224x224 (the size actually sent), averaged over frames."""
    imgs = [resize_224(img) for _, img in frames]
    print("\n=== Encode/decode latency @224x224 (median per image over "
          f"{len(imgs)} frames) ===")
    print(f"  {'codec':<14} {'encode ms':>10} {'decode ms':>10} {'round-trip ms':>14} {'bytes':>8}")
    rows = {}
    for name in ["PIL JPEG q90", "cv2 JPEG q90", "cv2 WebP q80"]:
        enc, dec = _enc(name), _dec(name)
        # encode each frame; collect a representative encoded buffer for decode timing
        bufs = [bytes(np.asarray(enc(im))) for im in imgs]
        e = float(np.median([_timeit(lambda im=im: enc(im)) for im in imgs]))
        d = float(np.median([_timeit(lambda b=b: dec(b)) for b in bufs]))
        size = float(np.mean([len(b) for b in bufs]))
        rows[name] = (e, d, size)
        print(f"  {name:<14} {e:>10.2f} {d:>10.2f} {e + d:>14.2f} {size:>8.0f}")
    return rows


def main():
    frames = collect_frames()
    print(f"Loaded {len(frames)} real DROID images from lerobot/droid_100")
    native_h, native_w = frames[0][1].shape[:2]
    print(f"Native decode resolution: {native_h}x{native_w}x3")

    for res_name, transform in [
        (f"NATIVE {native_h}x{native_w}", lambda x: x),
        ("INFERENCE 224x224 (actually sent)", resize_224),
    ]:
        raw_sizes, jpeg90, jpeg75, png, webp80 = [], [], [], [], []
        for _, img in frames:
            im = transform(img)
            raw_sizes.append(im.nbytes)
            jpeg90.append(encoded_bytes(im, "JPEG", quality=90))
            jpeg75.append(encoded_bytes(im, "JPEG", quality=75))
            png.append(encoded_bytes(im, "PNG", optimize=True))
            webp80.append(encoded_bytes(im, "WEBP", quality=80))
        summarize(res_name, raw_sizes, {
            "JPEG q=90": jpeg90, "JPEG q=75": jpeg75,
            "PNG (lossless)": png, "WebP q=80": webp80,
        })

    lat = benchmark_latency(frames)

    # Per-observation totals: DROID sends 2 camera images (base + wrist) per step.
    # End-to-end added latency = encode(2) + transmit(2) + decode(2). Raw has no codec cost.
    print("\n=== Per-observation network cost (2 cameras: base + wrist), at 224x224 ===")
    raw_b = 2 * 224 * 224 * 3
    jpeg_b = 2 * lat["cv2 JPEG q90"][2]
    cv2_codec_ms = 2 * (lat["cv2 JPEG q90"][0] + lat["cv2 JPEG q90"][1])  # encode+decode, both cams

    def link(mbit):  # one-way transmit ms for a payload of `b` bytes on an `mbit` Mbit/s link
        return lambda b: b * 8 / (mbit * 1e6) * 1e3

    LINKS = [("localhost ~10 Gb/s", 10000), ("1 GbE", 1000), ("WiFi ~50 Mb/s", 50), ("LTE ~10 Mb/s", 10)]
    print(f"  raw  = {raw_b:7.0f} B ({raw_b/1024:5.1f} KiB)   "
          f"JPEG q90 = {jpeg_b:6.0f} B ({jpeg_b/1024:.1f} KiB), {raw_b/jpeg_b:.1f}x smaller, "
          f"codec(enc+dec) = {cv2_codec_ms:.2f} ms (cv2/libjpeg-turbo)")
    print(f"\n  {'link':<20} {'raw transmit':>13} {'JPEG transmit':>15} {'JPEG+codec':>12} {'winner':>10}")
    for nm, mbit in LINKS:
        tx = link(mbit)
        raw_ms, jpeg_ms = tx(raw_b), tx(jpeg_b)
        jpeg_total = jpeg_ms + cv2_codec_ms
        win = "raw" if raw_ms <= jpeg_total else "JPEG"
        print(f"  {nm:<20} {raw_ms:>11.2f}ms {jpeg_ms:>13.2f}ms {jpeg_total:>10.2f}ms {win:>10}")
    print("\n  (JPEG wins when raw-transmit > JPEG-transmit + encode + decode.)")


if __name__ == "__main__":
    main()
