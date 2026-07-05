#!/bin/bash
# Sweep LEVEL_SCALE (K) for the deblocking filter across the bench footage.
# Run from the REPO ROOT (paths are root-relative, same convention as the other bench scripts):
#   bash bench/sweep_deblock.sh <outdir> <K> [clip:frames ...]
set -e
OUT="$1"; K="$2"; shift 2
mkdir -p "$OUT"

DEBLOCK_RS=reitero_video/reitero_video_common/src/deblock.rs
sed -i "s/const LEVEL_SCALE: f32 = [0-9.]*/const LEVEL_SCALE: f32 = $K/" "$DEBLOCK_RS"
grep "LEVEL_SCALE: f32" "$DEBLOCK_RS"

cargo build --profile fast-dev -p reitero_video_quality_test --bin ri-quality 2>&1 | tail -1

for spec in "$@"; do
  clip="${spec%%:*}"; frames="${spec##*:}"
  for pt in "70 65" "85 80" "95 90"; do
    set -- $pt; iq=$1; eq=$2
    rep="$OUT/K${K}_${clip}_q${iq}.log"
    if [ -f "$rep" ]; then continue; fi
    ./target/fast-dev/ri-quality run -i "bench/sequences/${clip}.y4m" \
      --max-frames "$frames" --search-range 31 --skip-threshold 3 \
      --intra-quality "$iq" --inter-quality "$eq" \
      --report-path "$rep" > /dev/null 2>&1
    grep -E "Bytes × DSSIM|PSNR-Y mean" "$rep" | tr '\n' ' '
    echo "<- K=$K $clip q$iq/$eq"
  done
done
