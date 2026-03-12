# reitero_video_tools

CLI for encoding and decoding [Reitero video](https://github.com/xhighway999/riv) (`.riv`) files.

## Install

```sh
cargo install reitero_video_tools
```

For maximum performance on your own machine, build with the `full-opt` profile instead
(enables LTO, `panic=abort`, and `target-cpu=native`):

```sh
cargo build --profile full-opt -p reitero_video_tools
```

## Commands

### encode

Encode a video file to `.riv` format. Requires ffmpeg on the system.

```sh
ri-cli encode --input input.mp4 --output output.riv
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--intra-quality` | 90 | I-frame quality (1–100) |
| `--inter-quality` | 35 | P-frame quality (1–100) |
| `--search-range` | 12 | Motion search range in pixels |
| `--skip-threshold` | 3 | Block skip SAD threshold |
| `--me-zero-mv-threshold` | 0 | Early termination for zero-MV search (0 = off) |
| `--me-predictor-threshold` | 0 | Early termination for predictor search (0 = off) |
| `--rdo-lambda-mult` | 0.49 | RDO lambda multiplier |
| `--max-frames` | 0 | Max frames to encode (0 = all) |

### decode

Decode a `.riv` file to an MP4, playback via mpv, a PPM sequence on stdout, or null (benchmark).

```sh
# Decode to MP4 (default)
ri-cli decode --input output.riv --output decoded.mp4

# Benchmark (decode and discard output)
ri-cli decode --input output.riv --mode null

# Pipe to mpv for playback (raw RGB24)
ri-cli decode --input output.riv --mode mpv

# Write PPM frames to stdout — pipe into ffmpeg or any image2pipe consumer
ri-cli decode --input output.riv --mode stdout | \
    ffmpeg -f image2pipe -vcodec ppm -i - output.mp4
```

The `stdout` mode writes a concatenated PPM sequence (`P6` binary PPM, one header + RGB24 pixel
data per frame). This is a no-dependency lossless path: no YUV conversion, no ffmpeg required to
decode. Progress and info messages go to stderr so they don't corrupt the stream.

Options:

| Flag | Description |
|---|---|
| `--mode` | Output mode: `file` (default), `null`, `mpv`, or `stdout` |
| `--output` | Output path (required for `file` mode) |
| `--skip-residuals` | Decode motion vectors only, skip residuals |
| `--instrument` | Print internal timing breakdown after decoding |

### extract-frame

Decode up to frame N and dump its internal components into `./reconstructed/frame_N/`.
Useful for inspecting codec internals.

```sh
ri-cli extract-frame --input output.riv --index 42
```

Output files:

| File | Description |
|---|---|
| `reconstructed.ppm` | Final decoded frame (RGB24, PPM format) |
| `predicted.ppm` | Motion-compensated prediction before residuals (inter frames only; same as reconstructed for intra) |
| `intra_residual.bin` | Raw RANS-compressed residual data (intra frames only) |
| `delta.yuv420` | Residual signal as raw little-endian i16 YUV420 (inter frames only) |

Note: all preceding frames must be decoded to reach frame N, so this is slow on large files.
