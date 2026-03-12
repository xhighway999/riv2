#!/usr/bin/env python3
"""Plot PSNR/SSIM vs bitrate curves from benchmark results."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BENCH_DIR = Path(__file__).parent

CODEC_STYLE = {
    #            label       color       lw     ls          marker  ms   zorder
    "riv2":  ("RIV",     "#E63946",  2.5,  "-",        "o",    7,   5),
    "vp9":   ("VP9",     "#4361EE",  1.5,  "--",       "s",    5,   4),
    "mpeg2": ("MPEG-2",  "#F4A261",  1.5,  (0,(5,2)),  "D",    5,   3),
    "mpeg1": ("MPEG-1",  "#2EC4B6",  1.5,  (0,(5,2)),  "^",    5,   3),
}

# Vertical reference lines: (label, bitrate_kbps)
BITRATE_REFS = [
    ("VCD",     1_150),
    ("DVD",     6_000),
    ("Blu-ray", 36_000),
]

QUALITY_REFS = {
    "psnr": [
        (25.0, "watchable"),
        (30.0, "acceptable"),
        (35.0, "good"),
        (40.0, "transparent"),
    ],
    "ssim": [
        (0.80, "watchable"),
        (0.85, "acceptable"),
        (0.95, "good"),
        (0.98, "transparent"),
    ],
}


def _load_deduped(seq_name: str) -> dict | None:
    results_path = BENCH_DIR / f"results_{seq_name}.json"
    if not results_path.exists():
        print(f"No results file: {results_path}", file=sys.stderr)
        return None
    data = json.loads(results_path.read_text())
    seen = set()
    deduped = []
    for r in data:
        key = (r["codec"], str(r["quality_param"]))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    codecs: dict[str, list] = {}
    for r in deduped:
        codecs.setdefault(r["codec"], []).append(r)
    return codecs



def _plot_metric(seq_name: str, metric: str, ylabel: str, codecs: dict, min_points: int = 3) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    codec_order = list(CODEC_STYLE.keys())
    for codec, points in sorted(codecs.items(), key=lambda x: codec_order.index(x[0]) if x[0] in codec_order else 99):
        style = CODEC_STYLE.get(codec)
        if style is None:
            continue
        label_base, color, lw, ls, marker, ms, zorder = style

        points = sorted(points, key=lambda r: r["bitrate_kbps"])
        br = np.array([r["bitrate_kbps"] for r in points])
        vals = np.array([r[metric] for r in points])

        partial = len(points) < min_points
        label = f"{label_base} (partial)" if partial else label_base

        ax.plot(
            br, vals,
            color=color, lw=lw, linestyle=ls,
            label=label, zorder=zorder,
            alpha=0.9 if not partial else 0.5,
        )
        ax.scatter(br, vals, color=color, s=ms**2, zorder=zorder + 1, edgecolors="white", linewidths=0.5)

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 5], numticks=10))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    seq_display = seq_name.replace("_", " ").title()
    ax.set_title(f"{seq_display} — {ylabel} vs Bitrate", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Bitrate (kbps)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x / 8_000, lambda x: x * 8_000),
    )
    secax.set_xscale("log")
    secax.set_xlabel("Bitrate (MB/s)", fontsize=10)
    secax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 5], numticks=10))
    secax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:g}")
    )

    ax.legend(fontsize=9.5, framealpha=0.9, loc="lower right")
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    # Quality reference lines
    if metric in QUALITY_REFS:
        ymin, ymax = ax.get_ylim()
        for ref_val, ref_label in QUALITY_REFS[metric]:
            if ymin <= ref_val <= ymax:
                ax.axhline(ref_val, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=1)
                ax.text(
                    ax.get_xlim()[0], ref_val,
                    f" {ref_label}", fontsize=7.5, color="gray",
                    va="bottom", ha="left",
                )

    # Bitrate reference lines — only draw if within the current x-axis range
    xmin, xmax = ax.get_xlim()
    for ref_label, ref_kbps in BITRATE_REFS:
        if xmin <= ref_kbps <= xmax:
            ax.axvline(ref_kbps, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=1)
            ax.text(
                ref_kbps, ax.get_ylim()[1],
                f" {ref_label}", fontsize=7.5, color="gray",
                va="top", ha="left", rotation=90,
            )

    plt.tight_layout()
    out = BENCH_DIR / f"{metric}_{seq_name}.png"
    plt.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close()


def plot_sequence(seq_name: str) -> None:
    codecs = _load_deduped(seq_name)
    if codecs is None:
        return
    _plot_metric(seq_name, "psnr", "PSNR (dB)", codecs)
    _plot_metric(seq_name, "ssim", "Perceived Quality (SSIM, 0–1)", codecs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", nargs="*", help="Sequence names (default: all available)")
    args = parser.parse_args()

    if args.sequences:
        names = [s.removesuffix(".y4m") for s in args.sequences]
    else:
        names = [p.stem.removeprefix("results_") for p in sorted(BENCH_DIR.glob("results_*.json"))
                 if "speedbench" not in p.stem]

    for name in names:
        plot_sequence(name)


if __name__ == "__main__":
    main()
