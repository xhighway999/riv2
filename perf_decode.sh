#!/bin/bash

#preheat
cargo run --profile ri_cli_fast --bin ri-cli -- \
    decode --input ./peng100.riv --mode null


perf record --call-graph=dwarf -- cargo run --profile ri_cli_fast --bin ri-cli -- \
    decode --input ./peng100.riv --mode null

hotspot ./perf.data &