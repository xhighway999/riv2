# reitero_video_common

Common types and utilities shared across the [Reitero video](https://github.com/xhighway999/riv) codec crates.

**This crate is an implementation detail of `reitero_encode` and `reitero_decode`.
Most applications should use those higher-level crates directly.**

## Main types

| Type | Description |
|---|---|
| `VideoHeader` | Parsed file header (width, height, fps, frame count) |
| `FrameType` | I-frame (`Intra`) vs P-frame (`Inter`) discriminant |
| `PackedFrame` | Raw binary frame header as it appears on disk |
| `Yuv420Frame` | Planar YUV420 frame with RGB24 conversion |
| `MotionVector` | Per-block motion vector (integer + half-pixel offsets) |
| `build_predicted` | Motion-compensated prediction (SIMD-accelerated) |
| `RIV_MAGIC` / `RIV_VERSION` | Format identification constants |
