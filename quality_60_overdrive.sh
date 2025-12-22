#!/usr/bin/env bash

# Load Cargo environment if available
if [ -f "$HOME/.cargo/env" ]; then
  . "$HOME/.cargo/env"
fi

# Ensure libclang is discoverable for bindgen (ffmpeg-next)
if [ -d /usr/lib/llvm-19/lib ]; then
  export LIBCLANG_PATH=${LIBCLANG_PATH:-/usr/lib/llvm-19/lib}
  export LD_LIBRARY_PATH=/usr/lib/llvm-19/lib:${LD_LIBRARY_PATH}
fi

exec cargo run --profile overdrive -p reitero_video_quality_test --bin ri-quality -- \
    run -i ./peng.mp4 --max-frames 60 --per-frame