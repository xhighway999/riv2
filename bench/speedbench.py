#!/usr/bin/env python3
"""
Speed benchmark: encode + decode fps at matched quality points.

Configs are hardcoded from existing results_*.json to produce comparable
PSNR across codecs. No PSNR/SSIM is measured here — just wall-clock speed.

Usage:
    python speedbench.py                     # all sequences
    python speedbench.py --seq foreman_cif   # one sequence
    python speedbench.py --runs 5            # more timing runs for decode
    python speedbench.py --force             # re-encode even if file exists
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

BENCH_DIR = Path(__file__).parent
IMAGE_SEQUENCES_DIR = BENCH_DIR / "image_sequences"
WORK_DIR = BENCH_DIR / "work"
RI_CLI = ["cargo", "run", "--profile", "full-opt", "-p", "reitero_video_tools", "--bin", "ri-cli", "--"]

# ---------------------------------------------------------------------------
# Hardcoded quality-matched configs
# Source: results_foreman_cif.json and results_park_joy_1080p50.json
#
# foreman_cif — target ~30 dB PSNR:
#   riv2  (85/80) → 30.74 dB
#   mpeg1 q=12    → 30.09 dB
#   mpeg2 q=12    → 30.06 dB
#   vp9   crf=50  → 30.20 dB
#
# park_joy_1080p50 — target ~27 dB PSNR:
#   riv2  (85/80) → 27.33 dB
#   mpeg1 q=12    → 26.69 dB
#   mpeg2 q=12    → 26.66 dB
#   vp9   crf=45  → 27.01 dB
# ---------------------------------------------------------------------------

SPEED_CONFIGS: dict[str, list[dict]] = {
    "foreman_cif": [
        {"codec": "riv2",  "label": "riv2 iq=85 eq=80",  "params": {"iq": 85, "eq": 80},  "ref_psnr": 30.74},
        {"codec": "mpeg1", "label": "mpeg1 q=12",         "params": {"q": 12},              "ref_psnr": 30.09},
        {"codec": "mpeg2", "label": "mpeg2 q=12",         "params": {"q": 12},              "ref_psnr": 30.06},
        {"codec": "vp9",   "label": "vp9 crf=50",         "params": {"crf": 50},            "ref_psnr": 30.20},
    ],
    "park_joy_1080p50": [
        {"codec": "riv2",  "label": "riv2 iq=85 eq=80",  "params": {"iq": 85, "eq": 80},  "ref_psnr": 27.33},
        {"codec": "mpeg1", "label": "mpeg1 q=12",         "params": {"q": 12},              "ref_psnr": 26.69},
        {"codec": "mpeg2", "label": "mpeg2 q=12",         "params": {"q": 12},              "ref_psnr": 26.66},
        {"codec": "vp9",   "label": "vp9 crf=45",         "params": {"crf": 45},            "ref_psnr": 27.01},
    ],
}


# ---------------------------------------------------------------------------
# Encode commands — return (encoded_path, elapsed_s)
# ---------------------------------------------------------------------------

def encode_riv2(seq_dir: Path, out_path: Path, fps: str, n_frames: int, iq: int, eq: int) -> float:
    cmd = [
        *RI_CLI, "encode",
        "-i", str(seq_dir / "frame_%06d.png"),
        "--fps", fps,
        "--intra-quality", str(iq), "--inter-quality", str(eq),
        "--search-range", "31", "--skip-threshold", "3",
        "--max-frames", str(n_frames),
        "-o", str(out_path),
    ]
    return _run_timed(cmd)


def encode_mpeg1(seq_dir: Path, out_path: Path, fps: str, n_frames: int, q: int) -> float:
    cmd = [
        "ffmpeg", "-y",
        "-r", fps,
        "-i", str(seq_dir / "frame_%06d.png"),
        "-vframes", str(n_frames),
        "-c:v", "mpeg1video", "-q:v", str(q), "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    return _run_timed(cmd)


def encode_mpeg2(seq_dir: Path, out_path: Path, fps: str, n_frames: int, q: int) -> float:
    cmd = [
        "ffmpeg", "-y",
        "-r", fps,
        "-i", str(seq_dir / "frame_%06d.png"),
        "-vframes", str(n_frames),
        "-c:v", "mpeg2video", "-q:v", str(q), "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    return _run_timed(cmd)


def encode_vp9(seq_dir: Path, out_path: Path, fps: str, n_frames: int, crf: int) -> float:
    cmd = [
        "ffmpeg", "-y",
        "-r", fps,
        "-i", str(seq_dir / "frame_%06d.png"),
        "-vframes", str(n_frames),
        "-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    return _run_timed(cmd)


# ---------------------------------------------------------------------------
# Decode commands — pipe output to /dev/null, return elapsed_s
# ---------------------------------------------------------------------------

def decode_riv2(encoded: Path) -> float:
    cmd = [*RI_CLI, "decode", "-i", str(encoded), "--mode", "stdout"]
    return _run_timed(cmd, stdout=subprocess.DEVNULL)


def decode_ffmpeg(encoded: Path) -> float:
    cmd = [
        "ffmpeg", "-i", str(encoded),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    return _run_timed(cmd, stdout=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_timed(cmd: list, stdout=None) -> float:
    t0 = time.monotonic()
    try:
        subprocess.run(
            cmd, check=True,
            stdout=stdout,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n  FAILED: {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        raise
    return time.monotonic() - t0


def encode_one(cfg: dict, seq_dir: Path, out_path: Path, fps: str, n_frames: int) -> float:
    codec = cfg["codec"]
    p = cfg["params"]
    if codec == "riv2":
        return encode_riv2(seq_dir, out_path, fps, n_frames, p["iq"], p["eq"])
    elif codec == "mpeg1":
        return encode_mpeg1(seq_dir, out_path, fps, n_frames, p["q"])
    elif codec == "mpeg2":
        return encode_mpeg2(seq_dir, out_path, fps, n_frames, p["q"])
    elif codec == "vp9":
        return encode_vp9(seq_dir, out_path, fps, n_frames, p["crf"])
    else:
        raise ValueError(f"Unknown codec: {codec}")


def decode_one(cfg: dict, encoded: Path) -> float:
    if cfg["codec"] == "riv2":
        return decode_riv2(encoded)
    else:
        return decode_ffmpeg(encoded)


def encoded_ext(codec: str) -> str:
    return {"riv2": ".riv", "mpeg1": ".mpg", "mpeg2": ".mpg", "vp9": ".webm"}[codec]


def encoded_path(seq_name: str, cfg: dict) -> Path:
    label = cfg["label"].replace(" ", "_").replace("=", "")
    return WORK_DIR / seq_name / f"speed_{label}{encoded_ext(cfg['codec'])}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sequence(seq_name: str, n_runs: int, force: bool) -> list[dict]:
    cfgs = SPEED_CONFIGS.get(seq_name)
    if not cfgs:
        print(f"No config for sequence '{seq_name}'", file=sys.stderr)
        return []

    seq_dir = IMAGE_SEQUENCES_DIR / seq_name
    meta_path = seq_dir / "meta.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Run convert.py first.", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    n_frames = meta["frame_count"]
    fps = meta["fps_rational"]

    print(f"\n{'='*60}")
    print(f"  {seq_name}  ({meta['width']}x{meta['height']} @ {meta['fps']} fps, {n_frames} frames)")
    print(f"{'='*60}")

    rows = []
    for cfg in cfgs:
        out = encoded_path(seq_name, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)

        # --- Encode ---
        if not out.exists() or force:
            print(f"\n  [{cfg['label']}] encoding...", flush=True)
            enc_time = encode_one(cfg, seq_dir, out, fps, n_frames)
            enc_fps = n_frames / enc_time
            print(f"    encode: {enc_fps:.1f} fps  ({enc_time:.1f}s)")
        else:
            # Re-time a single encode pass regardless (the file is a throwaway anyway)
            print(f"\n  [{cfg['label']}] encode (cached, timing one pass)...", flush=True)
            enc_time = encode_one(cfg, seq_dir, out, fps, n_frames)
            enc_fps = n_frames / enc_time
            print(f"    encode: {enc_fps:.1f} fps  ({enc_time:.1f}s)")

        # --- Decode (multiple runs, take median) ---
        print(f"    decoding {n_runs}x for timing...", flush=True)
        dec_times = []
        for i in range(n_runs):
            t = decode_one(cfg, out)
            dec_times.append(t)
            print(f"      run {i+1}: {n_frames/t:.1f} fps", flush=True)
        dec_time = median(dec_times)
        dec_fps = n_frames / dec_time

        size_mb = out.stat().st_size / 1024 / 1024
        rows.append({
            "sequence": seq_name,
            "codec": cfg["label"],
            "ref_psnr_db": cfg["ref_psnr"],
            "encode_fps": round(enc_fps, 1),
            "decode_fps": round(dec_fps, 1),
            "encoded_mb": round(size_mb, 2),
        })

    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    print()
    header = f"{'Codec':<22}  {'Ref PSNR':>9}  {'Enc fps':>8}  {'Dec fps':>8}  {'Size MB':>8}"
    print(header)
    print("-" * len(header))

    current_seq = None
    for r in rows:
        if r["sequence"] != current_seq:
            current_seq = r["sequence"]
            print(f"\n  {current_seq}")
        print(
            f"  {r['codec']:<20}  {r['ref_psnr_db']:>8.2f}  "
            f"{r['encode_fps']:>8.1f}  {r['decode_fps']:>8.1f}  {r['encoded_mb']:>7.1f}M"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Speed benchmark at matched quality points")
    parser.add_argument("--seq", help="Sequence name (e.g. foreman_cif)")
    parser.add_argument("--runs", type=int, default=3, help="Decode timing runs per codec (default: 3)")
    parser.add_argument("--force", action="store_true", help="Re-encode even if output already exists")
    args = parser.parse_args()

    sequences = [args.seq] if args.seq else list(SPEED_CONFIGS.keys())

    all_rows = []
    for seq in sequences:
        all_rows.extend(run_sequence(seq, args.runs, args.force))

    print_table(all_rows)

    out_path = BENCH_DIR / "results_speedbench.json"
    out_path.write_text(json.dumps(all_rows, indent=2) + "\n")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
