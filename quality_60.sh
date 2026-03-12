cargo run --profile fast-dev -p reitero_video_quality_test --bin ri-quality -- \
    run -i ./bench/sequences/foreman_cif.y4m --max-frames 300 --per-frame \
    --search-range 31 --skip-threshold 3