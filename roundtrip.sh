cargo run --profile ri_cli_fast --bin ri-cli -- encode --input ./a.mp4 --output a.riv --intra-quality 85 --inter-quality 40 --skip-threshold 4
cargo run --profile ri_cli_fast  --bin ri-cli -- decode --input ./a.riv --output b.mp4 
