#!/usr/bin/env python3
"""Benchmark pipeline: encode/decode PNG sequences, measure PSNR/SSIM."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

BENCH_DIR = Path(__file__).parent
IMAGE_SEQUENCES_DIR = BENCH_DIR / "image_sequences"
WORK_DIR = BENCH_DIR / "work"
RI_CLI = ["cargo", "run", "--profile", "full-opt", "-p", "reitero_video_tools", "--bin", "ri-cli", "--"]

CODECS = {
    "mpeg1": {"quality_points": [2, 4, 6, 8, 12, 16, 20, 25]},
    "mpeg2": {"quality_points": [2, 4, 6, 8, 12, 16, 20, 25]},
    "divx":  {"quality_points": [2, 4, 6, 8, 12, 16, 20, 25]},
    "riv2":  {"quality_points": [(70,65),(75,70),(80,75),(85,80),(90,85),(95,90)]},
    "vp9":   {"quality_points": [15, 20, 25, 30, 35, 40, 45, 50]},
}
QUICK_CODECS = {
    "mpeg1": {"quality_points": [4, 8, 16]},
    "mpeg2": {"quality_points": [4, 8, 16]},
    "divx":  {"quality_points": [4, 8, 16]},
    "riv2":  {"quality_points": [(75,70),(85,80),(95,90)]},
    "vp9":   {"quality_points": [20, 30, 40]},
}


# ---------------------------------------------------------------------------
# Streaming PPM reader
# ---------------------------------------------------------------------------

def stream_ppm_frames(pipe) -> Iterator[np.ndarray]:
    """Yield HxWx3 uint8 arrays from a binary PPM stream, frame by frame."""
    def read_exact(n):
        buf = b""
        while len(buf) < n:
            chunk = pipe.read(n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return buf

    def read_line():
        line = b""
        while True:
            c = pipe.read(1)
            if not c or c == b"\n":
                return line
            line += c

    while True:
        try:
            magic = pipe.read(2)
        except Exception:
            break
        if not magic or magic != b"P6":
            break
        pipe.read(1)  # newline after P6
        w, h = map(int, read_line().split())
        read_line()  # maxval
        data = read_exact(w * h * 3)
        yield np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return structural_similarity(a, b, channel_axis=2, data_range=255)


def _score_frame(dec_frame: np.ndarray, ref_path: Path) -> tuple[float, float]:
    ref = np.array(Image.open(ref_path).convert("RGB"))
    return psnr(ref, dec_frame), ssim(ref, dec_frame)


def measure_streamed(ppm_pipe, ref_frames: list[Path]) -> tuple[float, float, int]:
    """Compare a live PPM stream against reference PNG paths. Returns (psnr, ssim, n).
    Pipe reading stays on main thread; scoring runs in a thread pool."""
    N_WORKERS = os.cpu_count() or 4
    IN_FLIGHT = 64
    total = len(ref_frames)
    t_start = time.monotonic()

    def _print_progress(done: int) -> None:
        pct = done / total * 100
        elapsed = time.monotonic() - t_start
        eta = (elapsed / done * (total - done)) if done > 0 else 0
        print(f"\r    scoring {done}/{total} ({pct:.0f}%)  ETA {eta:.0f}s   ", end="", flush=True)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = []
        for dec_frame, ref_path in zip(stream_ppm_frames(ppm_pipe), ref_frames):
            # If we have too many in-flight, drain the oldest before submitting more
            if len(futures) >= IN_FLIGHT:
                futures[0].result()  # blocks until oldest is done
                _print_progress(len(futures) - IN_FLIGHT + 1)
            futures.append(pool.submit(_score_frame, dec_frame.copy(), ref_path))

    print(f"\r    scoring {total}/{total} (100%)  {time.monotonic() - t_start:.1f}s elapsed   ")

    if not futures:
        raise RuntimeError("No frames decoded")

    psnr_sum = ssim_sum = 0.0
    for fut in futures:
        p, s = fut.result()
        psnr_sum += p
        ssim_sum += s
    n = len(futures)
    return psnr_sum / n, ssim_sum / n, n


# ---------------------------------------------------------------------------
# Encode helpers (write encoded file to disk, no decoded frames)
# ---------------------------------------------------------------------------

def run_cmd(cmd: list, dry_run: bool, label: str) -> tuple[float, subprocess.CompletedProcess | None]:
    cmd_str = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"  [dry-run] {label}:\n    {cmd_str}")
        return 0.0, None
    print(f"  {label}: {cmd_str}")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"\n  FAILED (exit {e.returncode}):", file=sys.stderr)
        if e.stderr:
            print(e.stderr.decode(errors="replace"), file=sys.stderr)
        raise
    return time.monotonic() - t0, proc


def encode_mpeg1(seq_dir: Path, out_dir: Path, fps_rational: str, n_frames: int, q: int, dry_run: bool) -> tuple[float, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = out_dir / "encoded.mpg"
    cmd = [
        "ffmpeg", "-y",
        "-r", fps_rational,
        "-i", str(seq_dir / "frame_%06d.png"),
        "-vframes", str(n_frames),
        "-c:v", "mpeg1video", "-q:v", str(q), "-pix_fmt", "yuv420p",
        str(encoded),
    ]
    elapsed, _ = run_cmd(cmd, dry_run, f"mpeg1 encode q={q}")
    return elapsed, encoded


def encode_mpeg2(seq_dir: Path, out_dir: Path, fps_rational: str, n_frames: int, q: int, dry_run: bool) -> tuple[float, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = out_dir / "encoded.mpg"
    cmd = [
        "ffmpeg", "-y",
        "-r", fps_rational,
        "-i", str(seq_dir / "frame_%06d.png"),
        "-vframes", str(n_frames),
        "-c:v", "mpeg2video", "-q:v", str(q), "-pix_fmt", "yuv420p",
        str(encoded),
    ]
    elapsed, _ = run_cmd(cmd, dry_run, f"mpeg2 encode q={q}")
    return elapsed, encoded


def encode_divx(seq_dir: Path, out_dir: Path, fps_rational: str, n_frames: int, q: int, dry_run: bool) -> tuple[float, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = out_dir / "encoded.avi"
    cmd = [
        "ffmpeg", "-y",
        "-r", fps_rational,
        "-i", str(seq_dir / "frame_%06d.png"),
        "-vframes", str(n_frames),
        "-c:v", "mpeg4", "-q:v", str(q), "-pix_fmt", "yuv420p",
        "-vtag", "DIVX",
        str(encoded),
    ]
    elapsed, _ = run_cmd(cmd, dry_run, f"divx encode q={q}")
    return elapsed, encoded


def encode_vp9(seq_dir: Path, out_dir: Path, fps_rational: str, n_frames: int, crf: int, dry_run: bool) -> tuple[float, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = out_dir / "encoded.webm"
    cmd = [
        "ffmpeg", "-y",
        "-r", fps_rational,
        "-i", str(seq_dir / "frame_%06d.png"),
        "-vframes", str(n_frames),
        "-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p",
        str(encoded),
    ]
    elapsed, _ = run_cmd(cmd, dry_run, f"vp9 encode crf={crf}")
    return elapsed, encoded


def encode_riv2(seq_dir: Path, out_dir: Path, fps_rational: str, n_frames: int, iq: int, eq: int, dry_run: bool) -> tuple[float, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = out_dir / "encoded.riv"
    cmd = [
        *RI_CLI, "encode",
        "-i", str(seq_dir / "frame_%06d.png"),
        "--fps", fps_rational,
        "--intra-quality", str(iq), "--inter-quality", str(eq),
        "--search-range", "31", "--skip-threshold", "3",
        "--max-frames", str(n_frames),
        "-o", str(encoded),
    ]
    elapsed, _ = run_cmd(cmd, dry_run, f"riv2 encode iq={iq} eq={eq}")
    return elapsed, encoded


# ---------------------------------------------------------------------------
# Decode + measure in one streaming pass (no intermediate files)
# ---------------------------------------------------------------------------

def decode_and_measure_mpeg1(encoded: Path, ref_frames: list[Path], dry_run: bool) -> tuple[float, float, float]:
    cmd = ["ffmpeg", "-i", str(encoded), "-f", "image2pipe", "-vcodec", "ppm", "pipe:1"]
    if dry_run:
        print(f"  [dry-run] mpeg1 decode+measure:\n    {' '.join(cmd)} | <psnr>")
        return 0.0, 0.0, 0.0
    print(f"  mpeg1 decode+measure: {' '.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    avg_psnr, avg_ssim, _ = measure_streamed(proc.stdout, ref_frames)
    proc.wait()
    return time.monotonic() - t0, avg_psnr, avg_ssim


def decode_and_measure_mpeg2(encoded: Path, ref_frames: list[Path], dry_run: bool) -> tuple[float, float, float]:
    cmd = ["ffmpeg", "-i", str(encoded), "-f", "image2pipe", "-vcodec", "ppm", "pipe:1"]
    if dry_run:
        print(f"  [dry-run] mpeg2 decode+measure:\n    {' '.join(cmd)} | <psnr>")
        return 0.0, 0.0, 0.0
    print(f"  mpeg2 decode+measure: {' '.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    avg_psnr, avg_ssim, _ = measure_streamed(proc.stdout, ref_frames)
    proc.wait()
    return time.monotonic() - t0, avg_psnr, avg_ssim


def decode_and_measure_divx(encoded: Path, ref_frames: list[Path], dry_run: bool) -> tuple[float, float, float]:
    cmd = ["ffmpeg", "-i", str(encoded), "-f", "image2pipe", "-vcodec", "ppm", "pipe:1"]
    if dry_run:
        print(f"  [dry-run] divx decode+measure:\n    {' '.join(cmd)} | <psnr>")
        return 0.0, 0.0, 0.0
    print(f"  divx decode+measure: {' '.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    avg_psnr, avg_ssim, _ = measure_streamed(proc.stdout, ref_frames)
    proc.wait()
    return time.monotonic() - t0, avg_psnr, avg_ssim


def decode_and_measure_vp9(encoded: Path, ref_frames: list[Path], dry_run: bool) -> tuple[float, float, float]:
    cmd = ["ffmpeg", "-i", str(encoded), "-f", "image2pipe", "-vcodec", "ppm", "pipe:1"]
    if dry_run:
        print(f"  [dry-run] vp9 decode+measure:\n    {' '.join(cmd)} | <psnr>")
        return 0.0, 0.0, 0.0
    print(f"  vp9 decode+measure: {' '.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    avg_psnr, avg_ssim, _ = measure_streamed(proc.stdout, ref_frames)
    proc.wait()
    return time.monotonic() - t0, avg_psnr, avg_ssim


def decode_and_measure_riv2(encoded: Path, ref_frames: list[Path], dry_run: bool) -> tuple[float, float, float]:
    cmd = [*RI_CLI, "decode", "-i", str(encoded), "--mode", "stdout"]
    if dry_run:
        print(f"  [dry-run] riv2 decode+measure:\n    {' '.join(cmd)} | <psnr>")
        return 0.0, 0.0, 0.0
    print(f"  riv2 decode+measure: {' '.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    avg_psnr, avg_ssim, _ = measure_streamed(proc.stdout, ref_frames)
    proc.wait()
    return time.monotonic() - t0, avg_psnr, avg_ssim


# ---------------------------------------------------------------------------
# Per-codec benchmark generators
# ---------------------------------------------------------------------------

def ref_frame_paths(seq_dir: Path, n: int) -> list[Path]:
    return [seq_dir / f"frame_{i:06d}.png" for i in range(1, n + 1)]


def already_done(results: list[dict], codec: str, quality_param) -> bool:
    return any(r.get("codec") == codec and r.get("quality_param") == quality_param
               for r in results)


def bench_mpeg1(seq_name: str, seq_dir: Path, meta: dict, quality_points: list[int], dry_run: bool, results: list[dict]):
    fps_rational = meta["fps_rational"]
    n_frames = meta["frame_count"]
    refs = ref_frame_paths(seq_dir, n_frames)
    duration_s = n_frames / meta["fps"]

    for q in quality_points:
        if already_done(results, "mpeg1", q):
            print(f"\n--- {seq_name} / mpeg1_q{q}: already done, skipping ---")
            continue
        out_dir = WORK_DIR / seq_name / f"mpeg1_q{q}"
        print(f"\n--- {seq_name} / mpeg1_q{q} ---")

        enc_time, encoded = encode_mpeg1(seq_dir, out_dir, fps_rational, n_frames, q, dry_run)
        dec_time, avg_psnr, avg_ssim = decode_and_measure_mpeg1(encoded, refs, dry_run)

        if dry_run:
            yield {"sequence": seq_name, "codec": "mpeg1", "quality_param": q, "dry_run": True}
            continue

        encoded_bytes = encoded.stat().st_size
        bitrate_kbps = (encoded_bytes * 8) / duration_s / 1000
        yield {
            "sequence": seq_name, "codec": "mpeg1", "quality_param": q,
            "bitrate_kbps": round(bitrate_kbps, 2),
            "psnr": round(avg_psnr, 4), "ssim": round(avg_ssim, 6),
            "encode_time_s": round(enc_time, 3), "decode_time_s": round(dec_time, 3),
            "encode_fps": round(n_frames / enc_time, 2) if enc_time > 0 else None,
            "decode_fps": round(n_frames / dec_time, 2) if dec_time > 0 else None,
            "encoded_bytes": encoded_bytes,
        }


def bench_mpeg2(seq_name: str, seq_dir: Path, meta: dict, quality_points: list[int], dry_run: bool, results: list[dict]):
    fps_rational = meta["fps_rational"]
    n_frames = meta["frame_count"]
    refs = ref_frame_paths(seq_dir, n_frames)
    duration_s = n_frames / meta["fps"]

    for q in quality_points:
        if already_done(results, "mpeg2", q):
            print(f"\n--- {seq_name} / mpeg2_q{q}: already done, skipping ---")
            continue
        out_dir = WORK_DIR / seq_name / f"mpeg2_q{q}"
        print(f"\n--- {seq_name} / mpeg2_q{q} ---")

        enc_time, encoded = encode_mpeg2(seq_dir, out_dir, fps_rational, n_frames, q, dry_run)
        dec_time, avg_psnr, avg_ssim = decode_and_measure_mpeg2(encoded, refs, dry_run)

        if dry_run:
            yield {"sequence": seq_name, "codec": "mpeg2", "quality_param": q, "dry_run": True}
            continue

        encoded_bytes = encoded.stat().st_size
        bitrate_kbps = (encoded_bytes * 8) / duration_s / 1000
        yield {
            "sequence": seq_name, "codec": "mpeg2", "quality_param": q,
            "bitrate_kbps": round(bitrate_kbps, 2),
            "psnr": round(avg_psnr, 4), "ssim": round(avg_ssim, 6),
            "encode_time_s": round(enc_time, 3), "decode_time_s": round(dec_time, 3),
            "encode_fps": round(n_frames / enc_time, 2) if enc_time > 0 else None,
            "decode_fps": round(n_frames / dec_time, 2) if dec_time > 0 else None,
            "encoded_bytes": encoded_bytes,
        }


def bench_divx(seq_name: str, seq_dir: Path, meta: dict, quality_points: list[int], dry_run: bool, results: list[dict]):
    fps_rational = meta["fps_rational"]
    n_frames = meta["frame_count"]
    refs = ref_frame_paths(seq_dir, n_frames)
    duration_s = n_frames / meta["fps"]

    for q in quality_points:
        if already_done(results, "divx", q):
            print(f"\n--- {seq_name} / divx_q{q}: already done, skipping ---")
            continue
        out_dir = WORK_DIR / seq_name / f"divx_q{q}"
        print(f"\n--- {seq_name} / divx_q{q} ---")

        enc_time, encoded = encode_divx(seq_dir, out_dir, fps_rational, n_frames, q, dry_run)
        dec_time, avg_psnr, avg_ssim = decode_and_measure_divx(encoded, refs, dry_run)

        if dry_run:
            yield {"sequence": seq_name, "codec": "divx", "quality_param": q, "dry_run": True}
            continue

        encoded_bytes = encoded.stat().st_size
        bitrate_kbps = (encoded_bytes * 8) / duration_s / 1000
        yield {
            "sequence": seq_name, "codec": "divx", "quality_param": q,
            "bitrate_kbps": round(bitrate_kbps, 2),
            "psnr": round(avg_psnr, 4), "ssim": round(avg_ssim, 6),
            "encode_time_s": round(enc_time, 3), "decode_time_s": round(dec_time, 3),
            "encode_fps": round(n_frames / enc_time, 2) if enc_time > 0 else None,
            "decode_fps": round(n_frames / dec_time, 2) if dec_time > 0 else None,
            "encoded_bytes": encoded_bytes,
        }


def bench_vp9(seq_name: str, seq_dir: Path, meta: dict, quality_points: list[int], dry_run: bool, results: list[dict]):
    fps_rational = meta["fps_rational"]
    n_frames = meta["frame_count"]
    refs = ref_frame_paths(seq_dir, n_frames)
    duration_s = n_frames / meta["fps"]

    for crf in quality_points:
        if already_done(results, "vp9", crf):
            print(f"\n--- {seq_name} / vp9_crf{crf}: already done, skipping ---")
            continue
        out_dir = WORK_DIR / seq_name / f"vp9_crf{crf}"
        print(f"\n--- {seq_name} / vp9_crf{crf} ---")

        enc_time, encoded = encode_vp9(seq_dir, out_dir, fps_rational, n_frames, crf, dry_run)
        dec_time, avg_psnr, avg_ssim = decode_and_measure_vp9(encoded, refs, dry_run)

        if dry_run:
            yield {"sequence": seq_name, "codec": "vp9", "quality_param": crf, "dry_run": True}
            continue

        encoded_bytes = encoded.stat().st_size
        bitrate_kbps = (encoded_bytes * 8) / duration_s / 1000
        yield {
            "sequence": seq_name, "codec": "vp9", "quality_param": crf,
            "bitrate_kbps": round(bitrate_kbps, 2),
            "psnr": round(avg_psnr, 4), "ssim": round(avg_ssim, 6),
            "encode_time_s": round(enc_time, 3), "decode_time_s": round(dec_time, 3),
            "encode_fps": round(n_frames / enc_time, 2) if enc_time > 0 else None,
            "decode_fps": round(n_frames / dec_time, 2) if dec_time > 0 else None,
            "encoded_bytes": encoded_bytes,
        }


def bench_riv2(seq_name: str, seq_dir: Path, meta: dict, quality_points: list[tuple], dry_run: bool, results: list[dict]):
    fps_rational = meta["fps_rational"]
    n_frames = meta["frame_count"]
    refs = ref_frame_paths(seq_dir, n_frames)
    duration_s = n_frames / meta["fps"]

    for iq, eq in quality_points:
        if already_done(results, "riv2", [iq, eq]):
            print(f"\n--- {seq_name} / riv2_q{iq}q{eq}: already done, skipping ---")
            continue
        out_dir = WORK_DIR / seq_name / f"riv2_q{iq}q{eq}"
        print(f"\n--- {seq_name} / riv2_q{iq}q{eq} ---")

        enc_time, encoded = encode_riv2(seq_dir, out_dir, fps_rational, n_frames, iq, eq, dry_run)
        dec_time, avg_psnr, avg_ssim = decode_and_measure_riv2(encoded, refs, dry_run)

        if dry_run:
            yield {"sequence": seq_name, "codec": "riv2", "quality_param": (iq, eq), "dry_run": True}
            continue

        encoded_bytes = encoded.stat().st_size
        bitrate_kbps = (encoded_bytes * 8) / duration_s / 1000
        yield {
            "sequence": seq_name, "codec": "riv2", "quality_param": [iq, eq],
            "bitrate_kbps": round(bitrate_kbps, 2),
            "psnr": round(avg_psnr, 4), "ssim": round(avg_ssim, 6),
            "encode_time_s": round(enc_time, 3), "decode_time_s": round(dec_time, 3),
            "encode_fps": round(n_frames / enc_time, 2) if enc_time > 0 else None,
            "decode_fps": round(n_frames / dec_time, 2) if dec_time > 0 else None,
            "encoded_bytes": encoded_bytes,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def save_results(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps(results, indent=2) + "\n")


def run_sequence(seq_name: str, codecs: dict, dry_run: bool) -> None:
    seq_dir = IMAGE_SEQUENCES_DIR / seq_name
    meta_path = seq_dir / "meta.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Run convert.py first.", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    results_path = BENCH_DIR / f"results_{seq_name}.json"
    results = load_results(results_path)

    for codec, cfg in codecs.items():
        if codec == "mpeg1":
            gen = bench_mpeg1(seq_name, seq_dir, meta, cfg["quality_points"], dry_run, results)
        elif codec == "mpeg2":
            gen = bench_mpeg2(seq_name, seq_dir, meta, cfg["quality_points"], dry_run, results)
        elif codec == "divx":
            gen = bench_divx(seq_name, seq_dir, meta, cfg["quality_points"], dry_run, results)
        elif codec == "riv2":
            gen = bench_riv2(seq_name, seq_dir, meta, cfg["quality_points"], dry_run, results)
        elif codec == "vp9":
            gen = bench_vp9(seq_name, seq_dir, meta, cfg["quality_points"], dry_run, results)
        else:
            print(f"Unknown codec: {codec}", file=sys.stderr)
            continue

        for point in gen:
            results.append(point)
            if not dry_run:
                save_results(results_path, results)
                print(f"  PSNR={point['psnr']:.2f} dB  SSIM={point['ssim']:.4f}  "
                      f"{point['bitrate_kbps']:.0f} kbps")

    if dry_run:
        print("\nDry run complete — no files written.")
    else:
        print(f"\nResults written to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark pipeline")
    parser.add_argument("--seq", help="Sequence name (e.g. foreman_cif.y4m or foreman_cif)")
    parser.add_argument("--quick", action="store_true", help="Use quick quality point subset")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--codec", help="Run only this codec (mpeg1, vp9, or riv2)")
    args = parser.parse_args()

    codecs = QUICK_CODECS if args.quick else CODECS
    if args.codec:
        codecs = {args.codec: codecs[args.codec]}

    if args.seq:
        run_sequence(args.seq.removesuffix(".y4m"), codecs, args.dry_run)
    else:
        for seq_dir in sorted(IMAGE_SEQUENCES_DIR.iterdir()):
            if seq_dir.is_dir() and (seq_dir / "meta.json").exists():
                run_sequence(seq_dir.name, codecs, args.dry_run)


if __name__ == "__main__":
    main()
