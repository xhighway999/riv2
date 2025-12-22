#!/usr/bin/env bash
set -euo pipefail
# This script runs the instrumented decoder and writes stdout/stderr to instr_runs/run.log
# Note: previous attempts created penginstrument.sh; per request we stop creating that file and
# use instr_runs for logging here.
cd "$(dirname "$0")"
mkdir -p instr_runs

echo "Starting ri-cli decode (instrument mode); output -> instr_runs/run.log"
# Run in the ri_cli_fast profile (uses target-cpu=native) which is appropriate for performance runs
cargo run --profile ri_cli_fast --bin ri-cli -- decode --input ./peng100.riv --mode null --instrument 2>&1 | tee instr_runs/run.log || true

echo "---- instr_runs dir contents ----"
ls -la instr_runs || true

# Also show any penginstrument directory if the binary still writes there
if [ -d penginstrument ]; then
  echo "---- penginstrument dir contents (produced by code) ----"
  ls -la penginstrument || true
fi
