#!/usr/bin/env bash
# Download standard test sequences for codec benchmarking.
# Run from the bench/ directory.
set -euo pipefail

SEQS_DIR="$(dirname "$0")/sequences"
mkdir -p "$SEQS_DIR"

download() {
    local url="$1"
    local dest="$SEQS_DIR/$(basename "$url")"
    if [[ -f "$dest" ]]; then
        echo "already have $(basename "$url"), skipping"
        return
    fi
    echo "downloading $(basename "$url") ..."
    curl -L --progress-bar -o "$dest" "$url"
}

# Classic talking head, CIF (352x288), low motion — codec comfort zone
download "https://media.xiph.org/video/derf/y4m/foreman_cif.y4m"

# Outdoor foliage + motion, 1080p50 — punishing texture and motion
# 1.4 GB — comment out and use 720p variant if bandwidth is a concern:
# download "https://media.xiph.org/video/derf/y4m/park_joy_720p50.y4m"
download "https://media.xiph.org/video/derf/y4m/park_joy_1080p50.y4m"

echo "done. sequences in $SEQS_DIR"
