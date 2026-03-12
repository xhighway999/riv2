# reitero_encode

Encoder for the [Reitero video format](https://github.com/xhighway999/riv).

## Usage

```toml
[dependencies]
reitero_encode = "0.1"
```

### Encode to an in-memory buffer

```rust
use reitero_encode::{Encoder, EncoderConfig, Frame, VecWriter};

let config = EncoderConfig::new(1920, 1080, 30)
    .with_intra_quality(90)   // I-frame quality, 1–100
    .with_inter_quality(35);  // P-frame quality, 1–100

let mut encoder = Encoder::new(config, VecWriter::new())?;

// Feed RGB24 frames in order. data must be width * height * 3 bytes.
let frame = Frame::new(rgb_bytes, 1920, 1080, timestamp_ms);
let stats = encoder.encode_frame(frame)?;
println!("frame {} — {} bytes", stats.frame_index, stats.total_bytes);

// Flush and retrieve the encoded bitstream.
let riv_bytes: Vec<u8> = encoder.finish()?;
```

### Encode to a file

Implement [`VideoWriter`] for your sink — anything with a `write_all` + `seek` interface:

```rust
use reitero_encode::{Encoder, EncoderConfig, VideoWriter};
use std::fs::File;
use std::io::{Seek, Write};

struct FileWriter(File);

impl VideoWriter for FileWriter {
    fn write_all(&mut self, data: &[u8]) -> std::io::Result<()> { self.0.write_all(data) }
    fn position(&mut self) -> std::io::Result<u64> { self.0.stream_position() }
    fn seek(&mut self, pos: u64) -> std::io::Result<()> { self.0.seek(std::io::SeekFrom::Start(pos)).map(|_| ()) }
    fn flush(&mut self) -> std::io::Result<()> { self.0.flush() }
}
```

## Performance

Always build with `--release`. For maximum throughput on your own machine, enable
`target-cpu=native` via the workspace `full-opt` profile:

```sh
cargo build --profile full-opt
```

The encoder is the heavy path — motion search and DCT are the bottlenecks. The `simd`
feature (enabled by default) uses SIMD-accelerated motion estimation; disabling it
will significantly reduce encode speed.

## Configuration

| Option | Default | Description |
|---|---|---|
| `with_intra_quality(1–100)` | 90 | Quality of I-frames |
| `with_inter_quality(1–100)` | 35 | Quality of P-frames |
| `with_keyframe_interval(n)` | Fixed(30) | I-frame every N frames |
| `with_search_range(n)` | 12 | Motion search range in pixels |
| `with_skip_threshold(n)` | 3 | SAD threshold for skipping blocks |
| `with_rdo_lambda_mult(f)` | 0.49 | RDO lambda multiplier |
