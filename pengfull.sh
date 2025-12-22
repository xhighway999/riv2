unset argv0

cargo run --profile ri_cli_fast --bin ri-cli -- encode --input ./peng.mp4 --output peng100.riv --intra-quality 85 --inter-quality 80 --skip-threshold 12 --max-frames 100
cargo run --profile ri_cli_fast  --bin ri-cli -- decode --input ./peng100.riv --output b.mp4
