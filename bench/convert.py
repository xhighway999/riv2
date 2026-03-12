#!/usr/bin/env python3
"""Convert Y4M sequences to PNG image sequences with metadata."""

import json
import subprocess
import sys
from pathlib import Path

SEQUENCES_DIR = Path(__file__).parent / "sequences"
IMAGE_SEQUENCES_DIR = Path(__file__).parent / "image_sequences"


def probe(y4m_path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(y4m_path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")

    r_frame_rate = stream["r_frame_rate"]  # e.g. "30000/1001"
    num, den = map(int, r_frame_rate.split("/"))
    fps = num / den

    # frame_count: prefer nb_frames, fall back to duration * fps
    if "nb_frames" in stream and stream["nb_frames"] not in ("N/A", ""):
        frame_count = int(stream["nb_frames"])
    else:
        duration = float(data["format"]["duration"])
        frame_count = round(duration * fps)

    return {
        "fps": round(fps, 6),
        "fps_rational": r_frame_rate,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": frame_count,
    }


def convert(y4m_path: Path) -> None:
    name = y4m_path.stem
    out_dir = IMAGE_SEQUENCES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {y4m_path.name} -> {out_dir}/")

    # Extract all frames as PNG
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(y4m_path),
            str(out_dir / "frame_%06d.png"),
        ],
        check=True,
    )

    meta = probe(y4m_path)
    # Verify frame count matches actual output
    actual = len(list(out_dir.glob("frame_*.png")))
    if actual != meta["frame_count"]:
        print(f"  Warning: probed {meta['frame_count']} frames but found {actual} PNGs; using {actual}")
        meta["frame_count"] = actual

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  {actual} frames, {meta['width']}x{meta['height']} @ {meta['fps']:.4f} fps")


def main() -> None:
    if len(sys.argv) > 1:
        targets = [SEQUENCES_DIR / a for a in sys.argv[1:]]
    else:
        targets = sorted(SEQUENCES_DIR.glob("*.y4m"))

    if not targets:
        print("No sequences found.")
        sys.exit(1)

    for t in targets:
        if not t.exists():
            print(f"Not found: {t}", file=sys.stderr)
            sys.exit(1)
        convert(t)


if __name__ == "__main__":
    main()
